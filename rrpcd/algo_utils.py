from typing import List, Tuple, TypeVar, Iterable

import numpy as np

T = TypeVar('T')


def _safe_list2column(listlike: List[Tuple]) -> np.ndarray:
    return np.array([tuple([0, 1, 2]), tuple([0, 1])] + listlike)[2:][:, None]


def _unique_idxs(i_xs: Iterable[int]) -> List[int]:
    idxs = list()
    appeared_i_xs = set(i_xs)
    for idx, i_x in enumerate(i_xs):
        if i_x in appeared_i_xs:
            appeared_i_xs.remove(i_x)
            idxs.append(idx)
    return idxs
