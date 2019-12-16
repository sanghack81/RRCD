import sys
import time
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from typing import Set, List, Tuple, FrozenSet, TypeVar, Optional, AbstractSet, Sequence

import numpy as np
from pyrcds.domain import SkItem, AttributeClass
from pyrcds.model import PRCM, UndirectedRDep as URDep, enumerate_rdeps, RelationalVariable as RVar, SymTriple, RelationalPath, RCM, terminal_set
from pyrcds.rcds import new_extend, canonical_unshielded_triples
from pyrcds.utils import group_by

from rrpcd.algo_utils import _safe_list2column, _unique_idxs
from rrpcd.orienter import Orienter, OrientationInformation
from rrpcd.rci_test import RCITester
from rrpcd.utils import multiplys, refine_with, shuffled, is_1to1, mul2, set_combinations, pick, with_default

T = TypeVar('T')


class RCMLearner:
    # TODO check required tests early, determine required number of collider or non-collider, based on sepset rule and orientation rule, make the algorithm short-circuited.
    # TODO e.g, 'full' with 'conservative' option does not require testing all possible cases. --
    # TODO e.g., 'majority' with 'conservative' option, count total number of tests and check whether the condition is satisfied.
    # TODO minimum_rows_for_test to filter out unreliable test

    @staticmethod
    def sepset_rules() -> Tuple[str, ...]:
        """ Rules for how a separating set is determined """
        return 'minimal', 'first', 'full'

    @staticmethod
    def orientation_rules() -> Tuple[str, ...]:
        return 'majority', 'conservative', 'percentage'

    def __init__(self, rci_tester: RCITester, max_rv_hops: int, *,
                 max_degree: int = None, alpha=0.05, verbose=False, logfile=None, true_rcm=None, sepset_rule='minimal',
                 orientation_rule='majority', orientation_percentage_threshold=None, minimum_rows_for_test=0,
                 aggregator=None, detect_rbo_violations=True, detect_post_rbo_violations=True,
                 phase_I_order_independence=True):
        self.rci_tester = rci_tester
        self.kern = self.rci_tester.kern
        self.datasource = self.rci_tester.datasource
        self.fetcher = self.datasource.fetcher
        self.skeleton = rci_tester.skeleton
        self.schema = self.skeleton.schema

        # Parameters
        self.max_rv_hops = max_rv_hops
        self.max_degree = max_degree  # maximum number of causes per effect
        self.alpha = alpha
        self.verbose = verbose
        self.logfile = logfile
        self.true_rcm = true_rcm  # type: RCM
        self.sepset_rule = sepset_rule
        self.orientation_rule = orientation_rule
        self.orientation_percentage_threshold = orientation_percentage_threshold
        self.minimum_rows_for_test = minimum_rows_for_test
        self.aggregator = aggregator
        self.detect_rbo_violations = detect_rbo_violations
        self.detect_post_rbo_violations = detect_post_rbo_violations
        self.phase_I_order_independence = phase_I_order_independence

        # intermediate results
        self.prcm = PRCM(self.schema, set(URDep(dep) for dep in enumerate_rdeps(self.schema, max_rv_hops)))
        # CI test related information
        # key is of the format (cause, effect). separating sets are recorded for any CI query
        self.sepsets = defaultdict(lambda: None)
        # orientation-related information
        self.orientation_info = OrientationInformation(self.orientation_rule, self.orientation_percentage_threshold)

        # stats
        self.rbo_stats = defaultdict(lambda: 0)
        self.post_rbo_stats = defaultdict(lambda: 0)
        self.saved_by_aggregated_ci = set()

        self.__is_ci = lru_cache(maxsize=None)(self.__is_ci)
        self.__aggregated_ci = lru_cache(maxsize=None)(self.__aggregated_ci)
        self.__last_log_mode = None

        self.reached = lru_cache(maxsize=None)(self.reached)

    def reset_orientations(self, reset_prcm=True, reset_orientation_info=True):
        """ Remove """
        if reset_orientation_info:
            self.orientation_info = OrientationInformation(self.orientation_rule, self.orientation_percentage_threshold)
        if reset_prcm:
            self.prcm = PRCM(self.schema, self.prcm.undirected_dependencies | {URDep(d) for d in self.prcm.directed_dependencies})

    def reset_phase_I(self):
        """ Remove """
        self.prcm = PRCM(self.schema, set(URDep(dep) for dep in enumerate_rdeps(self.schema, self.max_rv_hops)))
        self.saved_by_aggregated_ci = set()

    def reached(self, P: RelationalPath, i):
        return tuple(sorted(terminal_set(self.skeleton, P, i)))

    def __log_to_remove_phase_I(self, udeps, padding=4):
        if not udeps:
            return
        self.__log('removing: ')
        cs, es = list(), list()
        for udep in udeps:
            c, e = next(iter(sorted(udep)))
            cs.append(str(c))
            es.append(str(e))
        max_len = max(len(c) for c in cs)
        cs = [' ' * (max_len - len(c)) + c for c in cs]
        for c, e in zip(cs, es):
            self.__log(padding * ' ', c, '_||_', e, sep=' ')
        self.__log('')

    def __log(self, *args, verbose=None, warning=False, **kwargs):
        if with_default(verbose, self.verbose):
            if self.__last_log_mode != warning:
                time.sleep(0.1)
            if warning:
                print(*args, file=sys.stderr, flush=True, **kwargs)
            else:
                print(*args, flush=True, **kwargs)

            if self.logfile:
                print(*args, file=self.logfile, flush=True, **kwargs)
        self.__last_log_mode = warning

    def is_ci(self, cause: RVar, effect: RVar, conds: AbstractSet[RVar], subset_size: int) -> bool:
        """ Perform multiple conditional independence tests given subsets of conditionals until a separating set is found """
        return self.__is_ci(cause, effect, frozenset(conds), subset_size)

    def __is_ci(self, cause: RVar, effect: RVar, conds: FrozenSet[RVar], subset_size: int) -> bool:
        assert 0 <= subset_size and effect.is_canonical

        for sub_conds in set_combinations(conds, subset_size):
            self.__log(f'                       {cause} _||_ {effect}' + (f' | {sub_conds}' if sub_conds else ''))
            ci_result = self.rci_tester(cause, effect, sub_conds)

            if ci_result.is_independent(self.alpha):
                if self.aggregator is None or self.aggregated_ci(cause, effect, sub_conds):
                    self.sepsets[(cause, effect)] = sub_conds
                    return True
                else:
                    self.saved_by_aggregated_ci.add((cause, effect, frozenset(conds)))

        return False

    def aggregated_ci(self, cause: RVar, effect: RVar, conds: AbstractSet[RVar]):
        return self.__aggregated_ci(cause, effect, frozenset(conds))

    def __aggregated_ci(self, cause: RVar, effect: RVar, conds: FrozenSet[RVar]):
        """ CI Test for f(cause) _||_ effect | conds """
        P, X = cause
        _, Y = effect

        """ (non-empty) Values """
        items_4_Y = self.fetcher.base_items[self.schema.attr2item_class[Y]]
        itemss_4_X = [self.reached(P, i) for i in items_4_Y]  # type: List[Tuple[SkItem]]

        selector = [i for i, items in enumerate(itemss_4_X) if items]
        items_4_Y, itemss_4_X = refine_with(selector, items_4_Y, itemss_4_X)

        """ Kernels """
        K_X = self.kernel_matrix(itemss_4_X, X, self.aggregator)
        K_Y = self.kernel_matrix(items_4_Y, Y)
        Ks = self.kern.KMs(conds, items_4_Y)
        K_cond = multiplys(*[Ks[cond] for cond in conds])

        """ Test """
        p_val = self.rci_tester.test_CI(K_X, K_Y, K_cond, p_value_only=True)
        return self.alpha < p_val

    def phase_I(self, *, verbose=None):
        """Find adjacencies of the underlying RCM."""
        self.reset_phase_I()

        udeps_remained = set(self.prcm.undirected_dependencies)
        rjust_len = max(len(str(c)) for u in udeps_remained for c, e in u) + 2

        def adj(x):
            return self.prcm.adj(x)

        # increasing depth (# of conditionals)
        depth = 0
        while udeps_remained and (self.max_degree is None or depth <= self.max_degree):
            self.__log(f'\nPHASE I: depth: {depth}', verbose=verbose)
            # CI tests and record undirected dependencies to be removed.
            to_remove = set()
            for udep in (sorted(udeps_remained) if self.phase_I_order_independence else shuffled(udeps_remained)):
                for cause, effect in (
                        sorted(udep) if self.phase_I_order_independence else shuffled(udep)):  # for both directions
                    self.__log(f'... {str(cause).rjust(rjust_len)} -- {effect}', verbose=verbose)
                    if self.is_ci(cause, effect, adj(effect) - {cause}, depth):
                        if self.phase_I_order_independence:
                            to_remove.add(udep)
                        else:
                            self.__log_to_remove_phase_I({udep})
                            udeps_remained -= {udep}
                            self.prcm.remove(udep)
                        break
                    if cause.is_canonical and depth == 0:  # no asymmetry
                        break

            # 'per iteration' clean-up to ensure order-independence.
            if self.phase_I_order_independence:
                self.__log_to_remove_phase_I(to_remove)
                udeps_remained -= to_remove
                self.prcm.remove(to_remove)

            udeps_remained = set(filter(lambda u: any(len(adj(d.effect)) - 1 >= depth + 1 for d in u), udeps_remained))
            depth += 1

    def perfect_phase_I(self):
        self.reset_phase_I()

        self.sepsets = defaultdict(lambda: None)  # reset sepsets
        self.prcm = PRCM(self.schema, {URDep(d) for d in self.true_rcm.directed_dependencies})

    def kernel_matrix(self, items: Sequence, attr: AttributeClass, aggregator=None) -> Optional[np.ndarray]:
        if isinstance(items[0], SkItem):
            assert aggregator is None
            return self.kern.K_comp([item[attr] for item in items], simple=True)

        if aggregator is None:
            return self.kern.K_comp(_safe_list2column([tuple([k[attr] for k in ks]) for ks in items]), simple=False)
        else:
            return self.kern.K_comp([self.aggregator([item[attr] for item in _items]) for _items in items], simple=True)

    def RBO_based_tests(self, *, verbose=None):
        self.__log('\nRBO --------------------------------------------------', verbose=verbose)
        self.rbo_stats = defaultdict(lambda: 0)

        """ Perform two types of tests first """
        considered = self.split_RBO_tests(verbose=verbose)
        considered |= self.pair_RBO_tests(verbose=verbose)

        """ If no separating set is found for some cases, then mark undetermined explicitly. """
        # It might be the case, 0 orientation information is recorded regarding an RBO-able pair (X,Y)
        # In such a case, make it "undetermined" explicitly.
        for X, Y in considered:
            if frozenset({X, Y}) in self.orientation_info.undetermined():
                continue
            if {(X, Y), (Y, X)} & self.orientation_info.orientations():
                continue
            # enforce undetermined ...
            self.orientation_info.add_record((X, Y), True)
            self.orientation_info.add_record((X, Y), False)

        self.__log('------------------------------------------------------', verbose=verbose)

    def pair_RBO_tests(self, *, verbose=None):
        def crv(attr: AttributeClass) -> RVar:
            return RVar(self.schema.attr2item_class[attr], attr)

        considered = set()
        for Y in sorted(self.schema.attrs):
            for P_X, Q_Z in combinations(sorted(self.prcm.adj(crv(Y))), 2):
                (P, X), (Q, Z) = P_X, Q_Z
                if X != Z:
                    continue

                if is_1to1(P) and is_1to1(Q):
                    considered.add(frozenset({X, Y}))
                    self.__generic_CUT_replacement('RBO', P_X, Y, Q_Z, verbose)

        return considered

    def split_RBO_tests(self, *, verbose=None):
        """ Orientation tests for each dependency P.X -- V_Y where either P is many or reversed(P) is many."""

        def crv(attr: AttributeClass) -> RVar:
            return RVar(self.schema.attr2item_class[attr], attr)

        rjust_len = max(len(str(P_X)) for Y in self.schema.attrs for P_X in self.prcm.adj(crv(Y))) + 2
        ljust_len = max(len(str(crv(Y))) for Y in self.schema.attrs) + 2

        def uai2019(ys, xss, idxs):
            to_out = []
            for idx in idxs:
                y, xs = ys[idx], xss[idx]
                xs = tuple(xs)
                for i, x in enumerate(xs):
                    subxs = xs[:i] + xs[i + 1:]
                    to_out.append((y, x, subxs))
            return list(zip(*to_out))

        considered = set()

        def rbo_pairs():
            for Y in sorted(self.schema.attrs):
                for P_X in sorted(self.prcm.adj(crv(Y))):
                    if P_X.rpath.is_many:
                        yield P_X, Y

        for P_X, Y in rbo_pairs():
            self.rbo_stats['case'] += 1
            P, X = P_X  # type: RelationalPath, AttributeClass
            V_Y = crv(Y)
            tilde_P_Y = RVar(P.reverse(), Y)
            considered.add(frozenset({X, Y}))

            is_collider = self.true_rcm is not None and P_X in self.true_rcm.pa(V_Y)
            is_non_collider = self.true_rcm is not None and P_X in self.true_rcm.ch(V_Y)

            """ Prepare data & kernel matrices """

            def _get_items_():
                _items_4_Y = self.fetcher.base_items[self.schema.attr2item_class[Y]]  # np.ndarray of items
                temp_itemss_4_X = [self.reached(P, item) for item in _items_4_Y]  # type: List[Tuple[SkItem]]
                at_least_two = [idx for idx, iks in enumerate(temp_itemss_4_X) if len(iks) >= 2]
                ys, xs, xss = uai2019(_items_4_Y, temp_itemss_4_X, at_least_two)
                idxs = _unique_idxs(xs)
                return tuple(zip(*[(xs[idx], ys[idx], xss[idx]) for idx in idxs]))

            """ Kernel matrices """
            conds = self.prcm.adj(crv(X))

            K_X, K_Y, K_rest_X, K_agg_rest_X, Ks = dict(), dict(), dict(), dict(), dict()

            def _initialize_kernel_():
                items_4_X, items_4_Y, rest_itemss_4_X = _get_items_()

                K_X[True] = self.kernel_matrix(items_4_X, X)
                K_Y[True] = self.kernel_matrix(items_4_Y, Y)
                K_rest_X[True] = self.kernel_matrix(rest_itemss_4_X, X)
                K_agg_rest_X[True] = self.kernel_matrix(rest_itemss_4_X, X, self.aggregator) if self.aggregator is not None else None
                Ks[True] = self.kern.KMs(conds, items_4_X)

            self.__log(f'RBO:  testing {str(P_X).rjust(rjust_len)} -- {str(V_Y).ljust(ljust_len)}', verbose=verbose)

            def ci_test(subconds=tuple(), with_K_Y=False):
                subconds = frozenset(subconds)
                appendix = ''
                if with_K_Y:
                    assert tilde_P_Y not in subconds
                    appendix = '-with-Y'

                """ Can we test without initialize? """
                cached_tests = 0
                p_val = self.rci_tester.get_cached(('split-RBO' + appendix, P_X, Y, subconds))
                cached_tests += 1
                if p_val is not None and p_val > self.alpha and self.aggregator is not None:
                    p_val = self.rci_tester.get_cached(('agg-split-RBO' + appendix, P_X, Y, subconds))
                    cached_tests += 1

                if p_val is not None:
                    self.rbo_stats['test'] += cached_tests
                    return p_val

                """ If not, initialize! """
                if not K_X:
                    _initialize_kernel_()

                K_cond = multiplys(*[Ks[True][cond] for cond in subconds])
                if with_K_Y:
                    K_cond = mul2(K_cond, K_Y[True])

                self.rbo_stats['test'] += 1
                p_val = self.rci_tester.test_CI(K_rest_X[True], K_X[True], K_cond, p_value_only=True, cache_key=('RBO' + appendix, P_X, Y, subconds))

                if p_val > self.alpha and self.aggregator is not None:
                    self.rbo_stats['test'] += 1
                    p_val = self.rci_tester.test_CI(K_agg_rest_X[True], K_X[True], K_cond, p_value_only=True, cache_key=('agg-RBO' + appendix, P_X, Y, subconds))

                return p_val

            if self.detect_rbo_violations:
                p_val_empty, p_val_y = ci_test(), ci_test(with_K_Y=True)

                violated = self._detect_viol('RBO', (X, Y), p_val_empty, p_val_y, verbose, is_collider, is_non_collider,
                                             self.rbo_stats,
                                             f'{str(P_X).rjust(rjust_len)} -- {str(V_Y).ljust(ljust_len)}',
                                             f'{str(P_X).rjust(rjust_len)} <- {str(V_Y).ljust(ljust_len)}',
                                             f'{str(P_X).rjust(rjust_len)} -> {str(V_Y).ljust(ljust_len)}')
                if violated:
                    continue

            # Perform tests with varying conditionals
            def _inner_(subconds):
                if self.alpha >= ci_test(subconds):
                    return False, None

                collider = self._inner_routine((X, Y), tilde_P_Y, subconds, lambda: ci_test(subconds, with_K_Y=True),
                                               self.detect_rbo_violations, self.rbo_stats, is_collider, is_non_collider)
                return True, collider

            self._ci_with_subsets(conds, _inner_)

        return considered

    def __generic_CUT_replacement(self, label, P_X: RVar, Y: AttributeClass, Q_Z: RVar, verbose=None):
        if label == 'RBO':
            assert P_X.attr == Q_Z.attr
            stats = self.rbo_stats
            detect_violation = self.detect_rbo_violations
        elif label == 'post-RBO':
            assert P_X.attr != Q_Z.attr
            stats = self.post_rbo_stats
            detect_violation = self.detect_post_rbo_violations
        else:
            raise ValueError(f'unknown label: {label}')
        stats['case'] += 1

        def crv(attr: AttributeClass) -> RVar:
            return RVar(self.schema.attr2item_class[attr], attr)

        rjust_len = max(len(str(P_X)) for Y in self.schema.attrs for P_X in self.prcm.adj(crv(Y))) + 2
        center_len = max(len(str(crv(Y))) for Y in self.schema.attrs) + 2

        P, X = P_X
        Q, Z = Q_Z
        V_Y = crv(Y)
        tilde_P = P.reverse()
        tilde_Q = Q.reverse()
        tilde_Q_Y = RVar(tilde_Q, Y)
        tilde_P_Y = RVar(tilde_P, Y)

        if X == Z:
            assert label == 'RBO'
        assert is_1to1(Q)

        is_collider = self.true_rcm is not None and {P_X, Q_Z} <= self.true_rcm.pa(V_Y)
        is_non_collider = self.true_rcm is not None and not is_collider

        conds_Z = self.prcm.adj(crv(Z))
        conds_X = self.prcm.adj(crv(X))

        K_XX, K_X, K_Y, K_Z, Ks_Z, Ks_X = dict(), dict(), dict(), dict(), dict(), dict()
        K_agg_X = dict()
        lazy_initialized = [False]

        def yoyoyaya(items):
            nonempties_idxs = [i for i, items in enumerate(items) if len(items) != 0]
            return nonempties_idxs

        def _lazy_initialize_kernel_matrices_():
            if lazy_initialized[0]:
                return

            multi_4_X_PX, items_4_X_PX, items_4_Y_PX, items_4_Z_PX, multi_4_X_PZ, dummy, items_4_Y_PZ, items_4_Z_PZ = self.fetch_for_test(P_X, Y, Q_Z)

            K_X[False] = None
            K_XX[False] = self.kernel_matrix(multi_4_X_PZ, X)
            K_agg_X[False] = self.kernel_matrix(multi_4_X_PZ, X, self.aggregator) if self.aggregator is not None else None

            K_Y[False] = self.kernel_matrix(items_4_Y_PZ, Y)
            K_Z[False] = self.kernel_matrix(items_4_Z_PZ, Z)

            K_X[True] = self.kernel_matrix(items_4_X_PX, X)

            expanded = None
            if multi_4_X_PX is not None:
                xxx = yoyoyaya(multi_4_X_PX)
                xxx_multi_4_X_PX = [multi_4_X_PX[idx] for idx in xxx]
                if len(xxx) > 10:
                    expanded = np.ones((len(multi_4_X_PX), len(multi_4_X_PX)))
                    xxx_K_XX = self.kernel_matrix(xxx_multi_4_X_PX, X)
                    expanded[tuple(np.meshgrid(xxx, xxx))] = xxx_K_XX
                else:
                    pass

            K_XX[True] = expanded
            K_Y[True] = self.kernel_matrix(items_4_Y_PX, Y)
            K_Z[True] = self.kernel_matrix(items_4_Z_PX, Z)

            Ks_Z[False] = self.kern.KMs(conds_Z, items_4_Z_PZ)
            Ks_X[True] = self.kern.KMs(conds_X, items_4_X_PX)

            lazy_initialized[0] = True

        self.__log(f'{label}:  testing {str(P_X).rjust(rjust_len)} -- {str(V_Y).center(center_len)} -- {str(Q_Z).ljust(rjust_len)}', verbose=verbose)

        K_cond_cache = dict()

        def lazy_ci_test(subconds, with_x: bool, Y_coll=False, block_rest=False):
            subconds = frozenset(subconds)
            assert not Y_coll or ((tilde_P_Y if with_x else tilde_Q_Y) not in subconds)
            cache_key = (f'{label}', P_X, Y, Q_Z, with_x, frozenset(subconds), Y_coll)
            agg_cache_key = (f'agg-{label}', *cache_key[1:])

            """ Check whether we can use cached result """
            p_val = self.rci_tester.get_cached(cache_key)
            temp_count = 1
            if p_val is not None and p_val > self.alpha and self.aggregator is not None and (
                    Q.is_many if with_x else P.is_many):
                p_val = self.rci_tester.get_cached(agg_cache_key)
                temp_count += 1
            if p_val is not None:
                stats['test'] += temp_count
                return p_val

            """ Initialize kernel matrices """
            _lazy_initialize_kernel_matrices_()

            """ Prepare Kernel for conditionals """
            cond_key = (with_x, subconds, Y_coll, block_rest)
            if cond_key in K_cond_cache:
                K_cond = K_cond_cache[cond_key]
            else:
                K_cond = multiplys(*[(Ks_X[with_x] if with_x else Ks_Z[with_x])[cond] for cond in subconds])
                if Y_coll:
                    K_cond = mul2(K_cond, K_Y[with_x])
                if block_rest and K_XX[True] is not None:
                    assert with_x
                    K_cond = mul2(K_cond, K_XX[with_x])
                K_cond_cache[cond_key] = K_cond

            kx = K_X[with_x] if with_x else K_XX[with_x]
            p_val = self.rci_tester.test_CI(kx, K_Z[with_x], K_cond, p_value_only=True, cache_key=cache_key)
            stats['test'] += 1
            if p_val > self.alpha and self.aggregator is not None and (Q.is_many if with_x else P.is_many):
                kx = K_X[with_x] if with_x else K_agg_X[with_x]
                p_val = self.rci_tester.test_CI(kx, K_Z[with_x], K_cond, p_value_only=True, cache_key=agg_cache_key)
                stats['test'] += 1

            return p_val

        """ Detect violation -- any violation will be counted. """
        if detect_violation:
            violated = False
            for with_X in [True, False]:
                p_val_Y = lazy_ci_test([], with_X, True, with_X and P.is_many)  # UAI 2019
                p_val_empty = lazy_ci_test([], with_X, False, False)  # UAI 2019

                violated |= self._detect_viol(label, SymTriple(X, Y, Z) if X != Z else (X, Y), p_val_empty, p_val_Y,
                                              verbose, is_collider, is_non_collider, stats,
                                              f'{str(P_X).rjust(rjust_len)} -- {str(V_Y).center(center_len)} -- {str(Q_Z).ljust(rjust_len)}',
                                              f'{str(P_X).rjust(rjust_len)} -- {str(V_Y).center(center_len)} -- {str(Q_Z).ljust(rjust_len)}',
                                              f'{str(P_X).rjust(rjust_len)} -> {str(V_Y).center(center_len)} <- {str(Q_Z).ljust(rjust_len)}')
                if is_1to1(P) and is_1to1(Q):
                    break

            if violated:
                return

        def _inner_(subconds, index=0):
            with_x = index == 1

            p_val = lazy_ci_test(subconds, with_x, False, False)
            if self.alpha >= p_val:
                return False, None

            rvar_Y = (tilde_P_Y if with_x else tilde_Q_Y)
            collider = self._inner_routine(SymTriple(X, Y, Z) if X != Z else (X, Y), rvar_Y, subconds,
                                           lambda: lazy_ci_test(subconds, with_x, True),
                                           detect_violation, stats, is_collider, is_non_collider)
            return True, collider

        # If RBO case, still two flattened representations are different although separating set should exist in either case
        # Hence, only when both P and Q is one where two flattened representations are the same, ... still they are different test?

        if X == Z:
            assert is_1to1(P) and is_1to1(Q)
            self._ci_with_subsets(conds_X, _inner_)  # mix adj of P_X and Q_Z together
        else:
            self._ci_with_two_subsets(conds_Z, conds_X, _inner_)  # mix adj of P_X and Q_Z together

    def post_RBO_unshielded_triples_tests(self, *, verbose=None, explicit_undetermined=True, skip_inference=False):
        def crv(attr: AttributeClass) -> RVar:
            return RVar(self.schema.attr2item_class[attr], attr)

        self.post_rbo_stats = defaultdict(lambda: 0)

        # self.orient()
        oriented = self.orientation_info.orientations()
        self.__log('\nPost-RBO ---------------------------------------------', verbose=verbose)

        considered = set()

        for Y in sorted(self.schema.attrs):
            for P_X, Q_Z in combinations(sorted(self.prcm.adj(crv(Y))), 2):
                (P, X), (Q, Z) = P_X, Q_Z
                if X == Z:
                    continue

                # shielded
                if all(RVar(R, X) in self.prcm.adj(crv(Z)) for R in new_extend(Q.reverse(), P)):
                    continue

                considered.add(SymTriple(X, Y, Z))

                # if any oriented, let it be X--Y, WLOG
                if {(Y, X), (X, Y), (Y, Z), (Z, Y)} & oriented:
                    if {(Y, Z), (Z, Y)} & oriented:
                        (P, X), (Q, Z) = P_X, Q_Z = Q_Z, P_X

                # both are *already* oriented
                if {(Y, X), (X, Y)} & oriented and {(Y, Z), (Z, Y)} & oriented:
                    if not skip_inference:
                        self.orientation_info.add_record(SymTriple(X, Y, Z), {(X, Y), (Z, Y)} <= oriented)
                    continue

                # non-collider (either shielded or unshielded)
                if (Y, X) in oriented:
                    if not skip_inference:
                        self.orientation_info.add_record(SymTriple(X, Y, Z), False)
                    continue

                # X --> Y -- Z
                if is_1to1(P) and is_1to1(Q):
                    self.__generic_CUT_replacement('post-RBO', P_X, Y, Q_Z, verbose)
                elif (X, Y) in oriented and is_1to1(Q):
                    self.__generic_CUT_replacement('post-RBO', P_X, Y, Q_Z, verbose)

        recorded = self.orientation_info.non_colliders() | self.orientation_info.colliders() | self.orientation_info.undetermined()
        if explicit_undetermined:
            for xyz in considered:
                if xyz not in recorded:
                    self.orientation_info.add_record(xyz, True)
                    self.orientation_info.add_record(xyz, False)

        self.__log('------------------------------------------------------', verbose=verbose)

    def CUT_based_collider_tests(self, verbose=None, rbo_only=False):
        def crv(attr: AttributeClass) -> RVar:
            return RVar(self.schema.attr2item_class[attr], attr)

        def cut_key(cut):
            _Vx, _, _Rz = cut
            assert _Vx != _Rz
            return tuple(sorted([_Vx, _Rz]))

        llen = max(len(str(crv(Y))) for Y in self.schema.attrs) + 2
        self.__log('\nCUT     ---------------------------------------------', verbose=verbose)
        for _, CUTs in group_by(list(canonical_unshielded_triples(self.prcm, single=False)), cut_key):
            Vx, _, Rz = next(iter(CUTs))
            x, z = Vx.attr, Rz.attr
            if rbo_only and x != z:
                continue

            candidate_attrs = {Py.attr for _, PPy, _ in CUTs for Py in PPy}
            self.__log(f'CUT:  testing {str(Vx).rjust(llen)} -- ? -- {Rz}', verbose=verbose)

            def _inner_(sub_conds):
                ci_result = self.rci_tester(Rz, Vx, sub_conds)
                if ci_result.is_independent(self.alpha):
                    sepset_attrs = {s.attr for s in sub_conds}
                    for y in candidate_attrs:
                        if x != z:
                            self.orientation_info.add_record(SymTriple(x, y, z), y not in sepset_attrs)
                        else:
                            if y not in sepset_attrs:
                                self.orientation_info.add_record((x, y), y not in sepset_attrs)
                            else:
                                self.orientation_info.add_record((y, x), y not in sepset_attrs)
                    return True, None
                return False, None

            self._ci_with_subsets(self.prcm.adj(Vx) - {Rz}, _inner_)
        self.__log('------------------------------------------------------', verbose=verbose)

    def orient(self):
        self.reset_orientations(reset_prcm=True, reset_orientation_info=False)

        pdag = Orienter(self.prcm.class_dependency_graph, self.orientation_info).orient()
        for x, y in pdag.oriented():
            self.prcm.orient_with(x, y)
        return

    def naive_orient(self):
        self.reset_orientations(reset_prcm=True, reset_orientation_info=False)

        pdag = Orienter(self.prcm.class_dependency_graph, self.orientation_info).sequential_orient()
        for x, y in pdag.oriented():
            self.prcm.orient_with(x, y)
        return

    def _ci_with_subsets(self, conditionals: Set, func):
        found = False
        for subset_size in range(len(conditionals) + 1):
            for sub_conditionals in set_combinations(conditionals, subset_size, randomized=self.sepset_rule == 'first'):
                sepset_found, _ = func(frozenset(sub_conditionals))
                found |= sepset_found
                if found and self.sepset_rule == 'first':
                    break
            if found and (self.sepset_rule == 'first' or self.sepset_rule == 'minimal'):
                break

    def _ci_with_two_subsets(self, conditionals0: Set, conditionals1: Set, func):
        found = False
        larger_size = max(len(conditionals0), len(conditionals1))
        for subset_size in range(larger_size + 1):
            first = np.random.randint(0, 2)

            for coin in [first, 1 - first]:
                index, conditionals = [(0, conditionals0), (1, conditionals1)][coin]
                if len(conditionals) >= subset_size:
                    for sub_conditionals in set_combinations(conditionals, subset_size,
                                                             randomized=self.sepset_rule == 'first'):
                        sepset_found, _ = func(frozenset(sub_conditionals), index)
                        found |= sepset_found
                        if found and self.sepset_rule == 'first':
                            break
                if found and self.sepset_rule == 'first':
                    break

            if found and (self.sepset_rule == 'first' or self.sepset_rule == 'minimal'):
                break

    def phase_II(self):
        self.RBO_based_tests()
        self.post_RBO_unshielded_triples_tests()
        self.orient()

    def fetch_for_test(self, P_X: RVar, Y: AttributeClass, Q_Z: RVar):
        assert is_1to1(Q_Z.rpath)
        P, X = P_X
        Q, Z = Q_Z

        items_4_Y = self.fetcher.base_items[self.schema.attr2item_class[Y]]
        multi_4_X = [self.reached(P, item) for item in items_4_Y]
        multi_4_Z = [self.reached(Q, item) for item in items_4_Y]
        selector = [idx for idx in range(len(items_4_Y)) if multi_4_X[idx] and multi_4_Z[idx]]
        multi_4_X, items_4_Y, multi_4_Z = refine_with(selector, multi_4_X, items_4_Y, multi_4_Z)

        # non-empty
        # with_x
        # with_z
        if X == Z:  # pair RBO
            assert not P.is_many
            items_4_Z = [pick(items) for items in multi_4_Z]
            items_4_X = [pick(items) for items in multi_4_X]
            selector = [idx for idx in range(len(items_4_X)) if items_4_X[idx] != items_4_Z[idx]]
            multi_4_X, items_4_X, items_4_Y, items_4_Z = refine_with(selector, multi_4_X, items_4_X, items_4_Y, items_4_Z)

            selector_for_X = _unique_idxs(items_4_X)
            multi_4_X_PX, items_4_X_PX, items_4_Y_PX, items_4_Z_PX = refine_with(selector_for_X, multi_4_X, items_4_X, items_4_Y, items_4_Z)

            selector_for_Z = _unique_idxs(items_4_Z)
            multi_4_X_PZ, items_4_X_PZ, items_4_Y_PZ, items_4_Z_PZ = refine_with(selector_for_Z, multi_4_X, items_4_X, items_4_Y, items_4_Z)

            return None, items_4_X_PX, items_4_Y_PX, items_4_Z_PX, multi_4_X_PZ, None, items_4_Y_PZ, items_4_Z_PZ
        else:

            items_4_Z_PZ = [pick(items) for items in multi_4_Z]
            selector_for_Z = _unique_idxs(items_4_Z_PZ)
            multi_4_X_PZ, items_4_Y_PZ, items_4_Z_PZ = refine_with(selector_for_Z, multi_4_X, items_4_Y, items_4_Z_PZ)

            restsxyz = []
            for x_items, y_item, z_items in zip(multi_4_X, items_4_Y, multi_4_Z):
                z_item = pick(z_items)
                x_items = tuple(x_items)
                for i, x_item in enumerate(x_items):
                    restsxyz.append(((x_items[:i] + x_items[i + 1:]), x_item, y_item, z_item))  # x,y,z,rest x

            multi_4_X_PX, items_4_X_PX, items_4_Y_PX, items_4_Z_PX = list(zip(*restsxyz))
            selector_for_X = _unique_idxs(items_4_X_PX)
            multi_4_X_PX, items_4_X_PX, items_4_Y_PX, items_4_Z_PX = refine_with(selector_for_X, multi_4_X_PX,
                                                                                 items_4_X_PX, items_4_Y_PX,
                                                                                 items_4_Z_PX)
            return multi_4_X_PX, items_4_X_PX, items_4_Y_PX, items_4_Z_PX, multi_4_X_PZ, None, items_4_Y_PZ, items_4_Z_PZ

    def _detect_viol(self, label, xy_or_xyz, p_val_empty, p_val_y, verbose=None, is_collider=False,
                     is_non_collider=False, stat_dict=None, log_weak='', log_vc='', log_vnc=''):
        """ Given two p values, determine whether two values are contradictory.
        Both cannot be independent.
        If one is independent, then we can infer a (non-)collider.
        """
        if p_val_empty > self.alpha and p_val_y > self.alpha:
            self.__log(f'{label}:     weak {log_weak}', verbose=verbose, warning=True)
            stat_dict['violation'] += 1
            return True

        if p_val_y > self.alpha:
            self.__log(f'{label}:  by viol {log_vc}', verbose=verbose, warning=is_collider)
            self.orientation_info.add_record(xy_or_xyz, False)
            stat_dict['violation-collider'] += 1
            stat_dict['correct violation-collider'] += int(is_non_collider)
            return True

        elif p_val_empty > self.alpha:
            self.__log(f'{label}:  by viol {log_vnc}', verbose=verbose, warning=is_non_collider)
            self.orientation_info.add_record(xy_or_xyz, True)
            stat_dict['violation-non-collider'] += 1  # violating non-collider assumption, hence collider
            stat_dict['correct violation-non-collider'] += int(is_collider)
            return True
        return False

    def _inner_routine(self, xy_or_xyz, rvar_Y, subconds, second_test, to_detect, stat_dict, is_collider,
                       is_non_collider):
        # independent...
        if rvar_Y not in subconds:
            if to_detect:
                if self.alpha >= second_test():
                    stat_dict['collider'] += 1
                    stat_dict['correct collider'] += int(is_collider)
                    self.orientation_info.add_record(xy_or_xyz, True)
                    return True
                else:
                    # by returning "True" in this case, it encourages fail fast
                    stat_dict['collider-fail'] += 1
                    stat_dict['correct collider-fail'] += int(is_non_collider)
                    return None
            else:
                stat_dict['collider'] += 1
                stat_dict['correct collider'] += int(is_collider)
                self.orientation_info.add_record(xy_or_xyz, True)
                return True
        else:
            stat_dict['non-collider'] += 1
            stat_dict['correct non-collider'] += int(is_non_collider)
            self.orientation_info.add_record(xy_or_xyz, False)
            return False
