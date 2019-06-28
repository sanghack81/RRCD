""" Perform a specific Relational Conditional Independence Test """

import collections
from functools import lru_cache
from typing import Set, FrozenSet

import numpy as np
from pyrcds.model import RelationalVariable as RVar
from sdcit.hsic import c_HSIC
from sdcit.sdcit import c_SDCIT

from rrpcd.data import purge_empty
from rrpcd.rel_kernel import RelationalKernelComputer
from rrpcd.utils import multiplys


class RCIResult(collections.namedtuple('RCIResult_', ['test_statistic', 'p_value', 'num_rows'])):
    def is_independent(self, alpha=0.05) -> bool:
        return self.p_value > alpha


class RCITester:
    """ Conditional Independence Test with Relational Variables"""

    def __init__(self, kern: RelationalKernelComputer, n_jobs=2):
        self.kern = kern
        self.datasource = kern.datasource
        self.skeleton = kern.datasource.skeleton
        self.n_jobs = n_jobs

        self.__cached_rci_test = lru_cache(maxsize=None)(self.__cached_rci_test)  # keep all results
        self.cached = dict()

    def rci_test(self, cause: RVar, effect: RVar, conds: Set[RVar], *, transform=None) -> RCIResult:
        """ RCI test with given relational variables """
        return self.__cached_rci_test(cause, effect, frozenset(conds), transform=transform)

    def __cached_rci_test(self, cause: RVar, effect: RVar, conds: FrozenSet[RVar], *, transform=None) -> RCIResult:
        """ Fetch & Purge & Sample """
        assert effect.is_canonical

        """ prepare kernel matrices """
        K_U, K_V, *K_Zs = [self.kern[v] for v in [cause, effect, *conds]]
        K_Z = multiplys(*K_Zs)

        """ remove empty `cause' column """
        cause_data, selector = purge_empty(self.datasource[cause], columns=(0,), return_selected=True)
        k_selector = np.ix_(selector, selector)

        if transform is not None:
            cause_data = transform(cause_data)
            K_U = self.kern.K_comp(cause_data, simple=transform.is_simple_transform)
            K_V, K_Z = K_V[k_selector], (K_Z[k_selector] if K_Z is not None else None)
        else:
            K_U, K_V, K_Z = K_U[k_selector], K_V[k_selector], (K_Z[k_selector] if K_Z is not None else None)

        test_statistic, p_value = self.test_CI(K_U, K_V, K_Z)

        return RCIResult(test_statistic, p_value, len(selector))

    def get_cached(self, cache_key):
        if cache_key is not None and cache_key in self.cached:
            return self.cached[cache_key]
        return None

    def test_CI(self, K_U, K_V, K_Z=None, *, num_nulls=500, p_value_only=False, cache_key=None):
        if cache_key is not None and cache_key in self.cached:
            return self.cached[cache_key]

        if K_Z is None:
            test_statistic, p_value = self.__hsic(K_U, K_V, num_nulls)
        else:
            test_statistic, p_value = self.__csdcit(K_U, K_V, K_Z, num_nulls)

        if cache_key is not None:
            self.cached[cache_key] = p_value if p_value_only else (test_statistic, p_value)

        if p_value_only:
            return p_value
        else:
            return test_statistic, p_value

    def __csdcit(self, K_U, K_V, K_Z, num_nulls):
        """ Call interface to c implementation"""
        if len(K_U) % 4:
            offset = len(K_U) % 4
            K_U, K_V, K_Z = K_U[offset:, offset:], K_V[offset:, offset:], K_Z[offset:, offset:]

        test_statistic, p_value, _ = c_SDCIT(K_U, K_V, K_Z,
                                             size_of_null_sample=num_nulls,
                                             with_null=True,
                                             n_jobs=self.n_jobs)
        return test_statistic, p_value

    def __hsic(self, K_U, K_V, num_nulls):
        test_statistic, p_value = c_HSIC(K_U, K_V,
                                         size_of_null_sample=num_nulls,
                                         n_jobs=self.n_jobs)
        return test_statistic, p_value

    def __call__(self, *args, **kwargs):
        return self.rci_test(*args, **kwargs)
