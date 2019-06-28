import numpy as np
from pyrcds.utils import median_except_diag


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
