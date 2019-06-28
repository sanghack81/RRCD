import numpy as np
from pyrcds.domain import generate_schema, generate_skeleton, remove_lone_entities
from pyrcds.model import enumerate_rvars
from pyrcds.utils import between_sampler

from rrpcd.data import purge_empty, DataCenter


def test_purge():
    x = np.array([(1, 2), (3,), (4, 5, 6), tuple(), (4, 5), (6, 7), tuple()])
    y = np.array([(1, 2), (3,), tuple(), (4, 5, 6), (4, 5), (6, 7), tuple()])
    z = np.array([(1, 2), (3,), tuple(), (4, 5, 6), tuple(), (6, 7), (1, 2, 3)])

    mat = np.vstack([x, y, z]).T

    ideal1 = mat[[0, 1, 4, 5], :]
    ideal2 = mat[[0, 1, 2, 4, 5], :]
    ideal3 = mat[[0, 1, 5], :]

    assert np.all(np.equal(purge_empty(mat), ideal1))
    assert np.all(np.equal(purge_empty(mat, columns=(0,)), ideal2))
    assert np.all(np.equal(purge_empty(mat, columns=(0, 1, 2)), ideal3))


def test_skdata():
    np.random.seed(1)
    schema = generate_schema(num_attr_classes_per_ent_class_distr=between_sampler(1, 1))
    skeleton = generate_skeleton(schema)
    for item in skeleton.items():
        for attr in item.item_class.attrs:
            item[attr] = np.random.randint(0, 10)

    remove_lone_entities(skeleton)
    dc = DataCenter(skeleton)
    for rvar in enumerate_rvars(schema, hop=5):
        print(rvar, len(purge_empty(dc[rvar], (0,))))
        # print(rvar, len(purge(vv, (0,))))

    import itertools
    for u, v, w in itertools.combinations(list(enumerate_rvars(schema, hop=5)), 3):
        if u.base == v.base == w.base:
            print(purge_empty(dc[(u, v, w)], (0, 1)))
            break
