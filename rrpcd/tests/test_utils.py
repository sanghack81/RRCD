import numpy as np
import pytest
from pyrcds.utils import median_except_diag

from rrpcd.algo_utils import _safe_list2column


@pytest.mark.parametrize('values', [[], [()], [(1,)], [(1, 2), (3, 4)], [(1,), (2, 3)]])
def test_tuple_column(values):
    column = _safe_list2column(values)
    assert column.shape == (len(values), 1)
    assert column.dtype == object
    assert column[:, 0].tolist() == values


def test_median_except_diag():
    D = np.array([[1, 3], [2, 5]])
    assert median_except_diag(D) == 2.5
    D = np.array([[1, 3, 4], [2, 4, 5], [4, 6, 6]])
    # 3,4,2,5,4,6
    assert median_except_diag(D) == 4
    D = np.array([[1, 3, 4], [2, 4, 5], [4, 100000, 6]])
    # 3,4,2,5,4,100000
    assert median_except_diag(D) == 4
    D = np.array([[1, 3, 4], [2, 4, 5], [5, 100000, 6]])
    # 3,4,2,5,5,100000
    assert median_except_diag(D) == 4.5
