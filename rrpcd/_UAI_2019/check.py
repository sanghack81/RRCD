from pyrcds.rcds import canonical_unshielded_triples
from pyrcds.tests.testing_utils import company_schema, company_rcm
from pyrcds.utils import group_by

if __name__ == '__main__':
    schema, rcm = company_schema(), company_rcm()


    def cut_key(cut):
        _Vx, _, _Rz = cut
        assert _Vx != _Rz
        return tuple(sorted([_Vx, _Rz]))


    for _, CUTs in group_by(list(canonical_unshielded_triples(rcm, single=False)), cut_key):
        for CUT in CUTs:
            print(CUT)
