""" Perform a specific Relational Conditional Independence Test """

import collections
from functools import lru_cache
from typing import Set, FrozenSet, Tuple, Union, Optional, Hashable, AbstractSet

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
    """ Conditional Independence Test with Relational Variables """

    def __init__(self, kern: RelationalKernelComputer, n_jobs: int = 2):
        """

        Parameters
        ----------
        kern : RelationalKernelComputer
            an interfact to obtain kernel matrices
        n_jobs : int
            a number of threads to be used to test conditional independence. For a large number of null samples, larger number is preferred. recommended 2 to 10 depending on the underlying system.
        """
        self.kern = kern
        self.datasource = kern.datasource
        self.skeleton = kern.datasource.skeleton
        self.n_jobs = n_jobs

        self.__cached_rci_test = lru_cache(maxsize=None)(self.__cached_rci_test)  # keep all results
        self.cached = dict()  # type: Dict[Hashable, Union[float, Tuple[float, float]]]

    def rci_test(self, cause: RVar, effect: RVar, conds: AbstractSet[RVar], *, transform=None) -> RCIResult:
        """ test RCI with given relational variables where effect must be canonical.

        Parameters
        ----------
        cause : RelationalVariable
        effect : RelationalVariable
        conds : Set[RelationalVariable]
        transform :

        Returns
        -------
        RCIResult which wraps a test statistic, p-value, and the number of data points (base items) involved.
        """
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

    def get_cached(self, cache_key) -> Optional[Union[float, Tuple[float, float]]]:
        if cache_key is not None and cache_key in self.cached:
            return self.cached[cache_key]
        return None

    def test_CI(self, K_U: np.ndarray, K_V: np.ndarray, K_Z: Optional[np.ndarray] = None, *, num_nulls: int = 500, p_value_only=False, cache_key=None) -> Union[float, Tuple[float, float]]:
        """ Test conditional independence U _||_ V | Z given corresponding kernel matrices taking advantage of cached results if `cache_key` is specified.

        Note that if the cached result is used, `p_value_only` is ignored and the previously used option will be applied.

        Parameters
        ----------
        K_U : np.ndarray
        K_V : np.ndarray
        K_Z : Optional[np.ndarray]
        num_nulls : int
            a number of null samples to generate to compute a p-value. the higher the better.
            Given a significance level of 0.05, more than 500 ~ 2000 is recommended.
        p_value_only : bool
            Whether to ignore test statistic
        cache_key :
            A cache key is an identifier for each unique CI test.
            Having the same key implies that a previous CI test result will be returned without checking the matrices are the same for the ones used in the previous test with the same key.

        Returns
        -------
        A test statistic from the underlying test and p-value or just p-vallue if `p_value_only` is `True`
        """
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

    def __csdcit(self, K_U: np.ndarray, K_V: np.ndarray, K_Z: np.ndarray, num_nulls) -> Tuple[float, float]:
        """ Call interface to a C implementation of SDCIT (conditional independence test) """
        if len(K_U) % 4:
            offset = len(K_U) % 4
            K_U, K_V, K_Z = K_U[offset:, offset:], K_V[offset:, offset:], K_Z[offset:, offset:]

        test_statistic, p_value, _ = c_SDCIT(K_U, K_V, K_Z,
                                             size_of_null_sample=num_nulls,
                                             with_null=True,
                                             n_jobs=self.n_jobs)
        return test_statistic, p_value

    def __hsic(self, K_U: np.ndarray, K_V: np.ndarray, num_nulls) -> Tuple[float, float]:
        """ Call interface to a C implementation of HSIC (marginal independence test) """
        test_statistic, p_value = c_HSIC(K_U, K_V,
                                         size_of_null_sample=num_nulls,
                                         n_jobs=self.n_jobs)
        return test_statistic, p_value

    def __call__(self, *args, **kwargs) -> Tuple[float, float]:
        """ Alias for `rci_test` """
        return self.rci_test(*args, **kwargs)
