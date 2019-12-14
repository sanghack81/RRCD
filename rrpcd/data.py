""" Handles data from skeleton (e.g., data cleaning, preprocessing, etc) """

from typing import Union, Iterable, Collection, Tuple

import numpy as np
from pyrcds.domain import RelationalSkeleton as RelSk, ImmutableRSkeleton
from pyrcds.model import RelationalVariable as RVar, SkeletonDataInterface


def purge_empty(mat: np.ndarray, columns: Tuple[int, ...] = (0, 1), return_selected=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Remove rows where cells are empty for specified columns"""
    if not columns:
        return mat
    vfunc = np.vectorize(lambda x: bool(x))
    submat = mat[:, columns]
    selected_rows = np.all(vfunc(submat), axis=1)
    if return_selected:
        return mat[selected_rows, :], selected_rows
    else:
        return mat[selected_rows, :]


class DataCenter:
    """ A class wrapping a skeleton and a data-fetching manager to serve as an abstraction layer for the underlying data implementation """

    def __init__(self, skeleton: RelSk):
        self.skeleton = ImmutableRSkeleton(skeleton)
        self.fetcher = SkeletonDataInterface(self.skeleton)

    def __getitem__(self, rvar_or_rvars: Union[RVar, Collection[RVar]]) -> np.ndarray:
        """ A 2d-matrix where each column corresponds to item attribute values where each cell is a tuple of values """
        if isinstance(rvar_or_rvars, RVar):
            return self.fetcher.inner_flatten(rvar_or_rvars, value_only=True, sort=False)[:, None]
        else:
            assert all([isinstance(rvar, RVar) for rvar in rvar_or_rvars])
            return np.hstack([self[rvar] for rvar in rvar_or_rvars])

    def simple_get(self, crv: RVar) -> np.ndarray:
        """ a column vector of item attribute values (where each cell is a value not a tuple of a single value) """
        return self.fetcher.fetch_singleton(crv)[:, None]
