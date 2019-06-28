import numpy as np
from numpy import abs
from pyrcds.domain import generate_skeleton
from pyrcds.model import RelationalPath, ParamRCM, generate_values_for_skeleton, normalize_skeleton
from pyrcds.model import RelationalVariable
from pyrcds.tests.testing_utils import company_rcm, company_schema
from pyrcds.utils import average_agg
from pyrcds.utils import linear_gaussian
from pyrcds.utils import normal_sampler

from rrpcd.rel_kernel import normalize_by_diag
from rrpcd.utils import multiplys


def test_normalize():
    for _ in range(10):
        x = abs(np.random.randn(10, 10)) + 1.0e-6
        x += x.T
        x = normalize_by_diag(x)
        for v in np.diag(x):
            assert abs(v - 1.0) <= 1.0e-6

    assert normalize_by_diag(None) is None


def test_multiply():
    x = abs(np.random.randn(10, 10)) + 1.0e-6
    y = abs(np.random.randn(10, 10)) + 1.0e-6
    z = abs(np.random.randn(10, 10)) + 1.0e-6

    assert np.allclose(multiplys(x, y, z), x * y * z)
    assert np.allclose(multiplys(x, y), x * y)
    assert np.allclose(multiplys(x), x)
    assert multiplys(None, None) is None
    aa = np.array([1])
    assert multiplys(None, aa, None, np.array([3])) == np.array([3])
    assert aa == np.array([1])
    print(aa)


def test_company():
    n = 400
    schema, rcm = company_schema(), company_rcm()
    functions = dict()
    effects = {RelationalVariable(RelationalPath(rcm.schema.item_class_of(attr)), attr) for attr in rcm.schema.attrs}
    skeleton = generate_skeleton(schema, n, max_degree=2)

    for e in effects:
        parameters = {cause: 1.0 for cause in rcm.pa(e)}
        functions[e] = linear_gaussian(parameters, average_agg(), normal_sampler(0, 0.3))

    rcm = ParamRCM(rcm.schema, rcm.directed_dependencies, functions)

    generate_values_for_skeleton(rcm, skeleton)
    normalize_skeleton(skeleton)
