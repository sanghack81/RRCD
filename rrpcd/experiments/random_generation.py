import json

import networkx as nx
import numpy as np
from pyrcds.domain import generate_schema, Cardinality
from pyrcds.model import generate_rcm, enumerate_rdeps
from pyrcds.rcds import rbos_colliders_non_colliders
from pyrcds.utils import ratio_sampler, between_sampler

if __name__ == '__main__':
    total_size = 1000
    np.random.seed(10)
    passed = 0
    schemas = list()
    rcms = list()
    while passed < total_size:
        print('.', end='', flush=True)
        schema = generate_schema(num_ent_classes_distr=ratio_sampler({3: 0.5, 4: 0.25, 5: 0.25}),
                                 num_rel_classes_distr=between_sampler(2, 5),
                                 num_ent_classes_per_rel_class_distr=ratio_sampler({2: 0.75, 3: 0.25}),
                                 num_attr_classes_per_ent_class_distr=between_sampler(1, 3),
                                 num_attr_classes_per_rel_class_distr=between_sampler(0, 1),
                                 cardinality_distr=ratio_sampler({Cardinality.many: 0.5, Cardinality.one: 0.5})  # Cardinality sampler
                                 )
        if set(schema.entity_classes) != {ec for rc in schema.relationship_classes for ec in rc.entity_classes}:
            continue
        if not nx.is_connected(schema.as_networkx_ug()):
            continue
        if len(schema.attrs) < 3:
            continue
        if len(schema.attrs) > 8:
            continue
        for _ in range(100):
            print(':', end='', flush=True)
            rcm = generate_rcm(schema=schema, num_dependencies=int(1.5 * len(schema.attrs)), max_degree=3, max_hop=between_sampler(2, 4).sample())
            if len(rcm.directed_dependencies) <= 2:
                continue
            cdg = rcm.class_dependency_graph
            if not nx.is_connected(cdg.as_networkx_dag().to_undirected()):
                continue
            if any(len(cdg.adj(attr)) == 0 for attr in schema.attrs):
                continue
            rbos, colliders, non_colliders = rbos_colliders_non_colliders(rcm)
            if len(rbos) + len(colliders) + len(non_colliders) == 0:
                continue
            break
        else:
            continue
        rcm_code = [at for at, dep in enumerate(sorted(list(enumerate_rdeps(schema, rcm.max_hop)))) if dep in rcm.directed_dependencies]

        schemas.append(schema.to_dict())
        rcms.append([rcm.max_hop, rcm_code])
        passed += 1
        print()

    with open(f'random/{total_size}_random_schemas.json', 'w') as f:
        json.dump(schemas, f, indent=4)
    with open(f'random/{total_size}_random_rcms.json', 'w') as f:
        json.dump(rcms, f, indent=4)

    with open(f'random/{total_size}_random_schemas.json', 'r') as f:
        schemas2 = json.load(f)
    with open(f'random/{total_size}_random_rcms.json', 'r') as f:
        rcms2 = json.load(f)

    assert schemas2 == schemas
    assert rcms2 == rcms
