from typing import List, Tuple, TypeVar, Sequence

import numpy as np

T = TypeVar('T')


def _safe_list2column(listlike: List[Tuple]) -> np.ndarray:
    """ a column vector corresponding to a list of tuples.

    If the sizes of tuples are the same, np.array may return a matrix where the number of columns corresponds to the common length of tuples.
    This method prevents such an unintended consequence.
    """
    return np.array([tuple([0, 1, 2]), tuple([0, 1])] + listlike)[2:][:, None]


def _unique_idxs(i_xs: Sequence[int]) -> List[int]:
    """ Returns unique values of a given sequence keeping the order based on when the value first appeared in the sequence """
    idxs = list()
    appeared_i_xs = set(i_xs)  # first refine
    for idx, i_x in enumerate(i_xs):
        if i_x in appeared_i_xs:
            appeared_i_xs.remove(i_x)
            idxs.append(idx)
    return idxs
