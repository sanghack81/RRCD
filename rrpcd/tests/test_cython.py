import numpy as np

from rrpcd.relkern.cy_set_dist import relational_kernel


def test_relational_kernel():
    # Item
    # D1 {0:0, 1:2}
    # D2 {1:0, 0:2, 2:3}
    # D3 {2:3, 1:0, 1:2}
    VK = np.array([[1, 0.5, 0.1],
                   [0.5, 1, 0.1],
                   [0.1, 0.1, 1]], order='C', dtype='float64')
    ones_VK = np.ones(VK.shape, order='C', dtype='float64')

    items = np.array([0, 1,
                      1, 0, 2,
                      2, 1, 1], order='C', dtype='int32')
    values = np.array([0., 2.,
                       0., 2., 3.,
                       3., 0., 2.], order='C', dtype='float64')
    lengths = np.array([2, 3, 3], order='C', dtype='int32')
    output = np.zeros((3, 3), order='C', dtype='float64')
    relational_kernel(values, items, lengths, ones_VK, output, 1, 1.0, 1, 1)
    K = output
    assert np.allclose(K, K.T)
    assert K[0, 1] == 0
    assert K[0, 2] == 0
    assert K[0, 0] == 2.0
    assert K[1, 1] == 3.0
    assert K[2, 2] == 3.0
    assert K[1, 2] == 3.0

    relational_kernel(values, items, lengths, ones_VK, output, 1, 1.0, 1, 0)
    K = output
    assert np.allclose(K, K.T)
    assert K[0, 0] == 2.0
    assert K[0, 1] == 2.0
    assert K[1, 1] == 3.0
    assert K[2, 2] == 3.0
    assert K[1, 2] == 3.0

    relational_kernel(values, items, lengths, VK, output, 1, 1.0, 1, 0)
    K = output
    assert np.allclose(K, K.T)
    assert K[0, 0] == 2.0
    assert K[1, 1] == 3.0
    assert K[2, 2] == 3.0
    assert K[1, 2] == 2.5
    assert K[2, 0] == 1.5

    relational_kernel(values, items, lengths, ones_VK, output, 1, 1.0, 0, 1)
    K = output
    assert K[0, 1] == 0
    assert K[0, 2] == 0
