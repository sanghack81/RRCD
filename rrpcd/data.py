""" Handles data from skeleton (e.g., data cleaning, preprocessing, etc) """

from typing import Union, Iterable

import numpy as np
from pyrcds.domain import RelationalSkeleton as RelSk, ImmutableRSkeleton
from pyrcds.model import RelationalVariable as RVar, SkeletonDataInterface


def purge_empty(mat, columns=(0, 1), return_selected=False):
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
    """ Wrap skeleton and data fetching manager """

    def __init__(self, skeleton: RelSk):
        self.skeleton = ImmutableRSkeleton(skeleton)
        self.fetcher = SkeletonDataInterface(self.skeleton)

    def __getitem__(self, rvar: Union[RVar, Iterable[RVar]]) -> np.ndarray:
        if isinstance(rvar, RVar):
            return self.fetcher.inner_flatten(rvar, value_only=True, sort=False)[:, None]
        else:
            assert all([isinstance(item, RVar) for item in rvar])
            return np.hstack([self[item] for item in rvar])

    def simple_get(self, crv: RVar) -> np.ndarray:
        return self.fetcher.fetch_singleton(crv)[:, None]
