import numpy as np
import pytest
from pyrcds.utils import median_except_diag

from rrpcd.utils import reproducible


def test_reproducible_restores_random_state_on_exception():
    @reproducible
    def interrupted():
        np.random.random(10)
        raise RuntimeError('interrupted experiment')

    np.random.seed(0)
    expected = np.random.random(3)
    np.random.seed(0)
    with pytest.raises(RuntimeError, match='interrupted experiment'):
        interrupted(seed=999)
    np.testing.assert_array_equal(np.random.random(3), expected)


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
