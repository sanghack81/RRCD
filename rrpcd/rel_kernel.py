import itertools
from functools import lru_cache
from typing import Optional, Dict, Iterable, AbstractSet, Sequence, TypeVar

import numpy as np
import numpy.ma as ma
from numpy import diag, zeros
from numpy.core.umath import sqrt
from pyrcds.model import enumerate_rvars, RelationalVariable as RV
from sklearn.metrics import euclidean_distances

from rrpcd.data import DataCenter
from rrpcd.relkern.cy_set_dist import relational_kernel00

T = TypeVar('T')


def normalize_by_diag(k: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """ Given a kernel matrix, normalize the matrix by its diagonal making its diagonal 1s and the rest of values normalized accordingly. """
    if k is None:
        return None

    diag_array = diag(k).copy()
    # TODO should we emit warnings?
    diag_array[diag_array == 0.] = 1.  # will divide by 1

    x = 1. / sqrt(np.repeat(diag_array[:, None], len(k), axis=1))
    k = (k * x) * x.transpose()
    return k


def relational_kernel_matrix00(data, n_jobs: int, gamma: float, equal_size_only=False):
    """ Simplified relational kernel matrix computation


    """

    def __unroll(list_of_lists: Sequence[Sequence[T]]) -> Sequence[T]:
        """ List of lists as a single list """
        return list(itertools.chain(*list_of_lists))

    unrolled = __unroll(data)
    value_data = np.array(unrolled, dtype='float64', order='C')
    lengths = np.array([len(d) for d in data], dtype='int32', order='C')
    n = len(lengths)
    output = zeros((n, n), dtype='float64', order='C')
    equal_size_only = 1 if equal_size_only else 0

    relational_kernel00(value_data, lengths, output, n_jobs, gamma, equal_size_only)

    return output


def rbf_gamma_median(x: np.ndarray, pass_D_squared=False):
    """ 1/(2*(median distance**2)) """
    assert len(x.shape) == 2
    D_squared = euclidean_distances(x, squared=True)
    # masking upper triangle and the diagonal.
    mask = np.triu(np.ones(D_squared.shape), 0)
    median_squared_distance = ma.median(ma.array(D_squared, mask=mask))
    if median_squared_distance == 0:
        xx = np.array(list(set(D_squared.reshape(D_squared.size))))  # a bit slow?
        if len(xx) > 1:
            xx = np.sort(xx)  # TODO better find 2nd smallest item... (next to 0)
            median_squared_distance = xx[1]
            assert median_squared_distance > 0
        else:
            median_squared_distance = 1

    if pass_D_squared:
        if median_squared_distance == 0 or np.isinf(median_squared_distance):
            return 0.5, D_squared
        return 0.5 / median_squared_distance, D_squared
    else:
        if median_squared_distance == 0 or np.isinf(median_squared_distance):
            return 0.5
        return 0.5 / median_squared_distance


def _extract_unique_values(tups) -> np.ndarray:
    """ Change tuples of values to an array of unique values, then make it a column """
    return np.array(list(set(list(itertools.chain(*tups)))))[:, None]


class RelationalKernelComputer:
    """ Computes Kernel matrices from underlying data source """

    def __init__(self, datasource: DataCenter):
        self.datasource = datasource
        self.fetcher = self.datasource.fetcher

    def K(self, rvar: RV):
        raise NotImplementedError()

    def K_comp(self, data, gamma=None, simple=False):
        raise NotImplementedError()

    def __getitem__(self, item):
        return self.K(item)

    def __call__(self, *args, **kwargs):
        return self.K(*args, **kwargs)

    def KMs(self, conditionals: AbstractSet[RV], items_of_interest: Iterable) -> Dict[RV, np.ndarray]:
        if not conditionals:
            return dict()
        x_base_items = self.fetcher.base_items[next(iter(conditionals)).base]
        # from x_base_items to is4x
        lookup = {x_b: idx for idx, x_b in enumerate(x_base_items)}
        cross_selector = [lookup[i_x] for i_x in items_of_interest]
        mesh = np.ix_(cross_selector, cross_selector)
        return {cond: self[cond][mesh] for cond in conditionals}


class RBFKernelComputer(RelationalKernelComputer):
    """ Compute a diagonal-normalized RBF kernel matrix with Gamma computed based on the median """

    def __init__(self, datasource: DataCenter, *, additive: float, n_jobs=1, eqsize_only=False, k_cache_max_size=32):
        """

        Parameters
        ----------
        datasource
        additive : float
            a small number to prevent zeros in resulting kernel matrix (TODO write detailed reasons)
        n_jobs : int
            a number of parallel jobs
        eqsize_only : bool
            Whether the similarity between two sets of different sizes should be 0
        k_cache_max_size
        """
        super().__init__(datasource)

        self.gamma = {crv.attr: rbf_gamma_median(self.datasource.simple_get(crv))
                      for crv in enumerate_rvars(datasource.skeleton.schema, 0)}

        self.additive = additive
        self.n_jobs = n_jobs
        self.eqsize_only = eqsize_only

        self.K = lru_cache(maxsize=k_cache_max_size)(self.K)  # for 1000x1000 approx 8MB

    def K(self, rvar: RV) -> np.ndarray:
        if rvar.is_canonical:
            data = self.datasource.simple_get(rvar)
            return self.K_comp(data, gamma_val=self.gamma[rvar.attr], simple=True)
        else:
            data = self.datasource[rvar]
            return self.K_comp(data, gamma_val=self.gamma[rvar.attr])

    def K_comp(self, data, gamma_val: float = None, simple: bool = False) -> np.ndarray:
        """ Kernel matrix given relational data (flattened) """
        if simple and not isinstance(data, np.ndarray):
            data = np.array(data)[:, None]

        if simple:
            if gamma_val is None:
                gamma_val, D_squared = rbf_gamma_median(data, pass_D_squared=True)
            else:
                D_squared = euclidean_distances(data, squared=True)

            G = np.exp(-gamma_val * D_squared)
        else:
            data = data.squeeze()
            if gamma_val is None:
                gamma_val = rbf_gamma_median(_extract_unique_values(data))  # unique values
            G = relational_kernel_matrix00(data, self.n_jobs, gamma_val, self.eqsize_only)

        # TODO additive?
        return normalize_by_diag((G + self.additive) / (1 + self.additive))
