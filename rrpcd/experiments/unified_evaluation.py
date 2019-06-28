import itertools
from collections import defaultdict
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from pyrcds.domain import RelationalSchema, SkItem, RelationalSkeleton

from rrpcd.experiments.exp_utils import files
from rrpcd.utils import reproducible


#
#
def stats_keys() -> List[str]:
    return ['case',
            'test',
            'violation',
            'violation-collider',
            'correct violation-collider',
            'violation-non-collider',
            'correct violation-non-collider',
            'collider',
            'correct collider',
            'non-collider',
            'correct non-collider',
            'collider-fail',
            'correct collider-fail',
            ]


@reproducible
def sized_random_skeleton(schema: RelationalSchema, sizing: dict) -> RelationalSkeleton:
    counter = itertools.count()

    def entity_generator(entity_class):
        while True:
            yield SkItem('e' + str(next(counter)), entity_class)

    def rel_generator(relationship_class):
        while True:
            yield SkItem('r' + str(next(counter)), relationship_class)

    item_generators = {**{ent_class: entity_generator(ent_class) for ent_class in schema.entities},
                       **{rel_class: rel_generator(rel_class) for rel_class in schema.relationships}}

    # build a graph first
    nodes = defaultdict(list)
    g = nx.Graph()
    for item_class in sorted(schema.item_classes):
        num_items = sizing[item_class]
        for _ in range(num_items):
            item = next(item_generators[item_class])
            nodes[item_class].append(item)
            g.add_node(item)

    for rel_class in sorted(schema.relationships):
        for ent_class in sorted(rel_class.entities):
            chosen_entities = np.random.choice(nodes[ent_class],
                                               size=len(nodes[rel_class]),
                                               replace=rel_class.is_many(ent_class))
            g.add_edges_from(zip(nodes[rel_class], chosen_entities))

    # translate the graph into a new skeleton
    skeleton = RelationalSkeleton(schema, strict=True)
    for ent_class in sorted(schema.entities):
        for e in nodes[ent_class]:
            skeleton.add_entity(e)
    for rel_class in sorted(schema.relationships):
        for r in nodes[rel_class]:
            skeleton.add_relationship(r, list(g.neighbors(r)))
    return skeleton


def sizing_method(base_size: int, schema: RelationalSchema):
    sizing = {ic: base_size for ic in schema.item_classes}
    for rc in schema.relationship_classes:
        all_many = all(rc.is_many(ec) for ec in rc.entity_classes)
        all_one = all(rc.is_many(ec) for ec in rc.entity_classes)
        if all_many:
            sizing[rc] = 2 * base_size
        elif all_one:
            for ec in rc.entity_classes:
                sizing[ec] = int(1.2 * sizing[ec])
    return sizing


def retrieve_finished(key_length: dict, working_dir: str) -> dict:
    done = defaultdict(set)
    for phase in [1, 2]:
        for fname in files(working_dir, prefix=f'phase_{phase}', suffix='.csv'):
            df = pd.read_csv(f'{working_dir}{fname}', header=None)
            for _, row in df.iloc[:, 0:key_length[phase]].iterrows():
                done[phase].add(tuple(row))
    return done

#
# if __name__ == '__main__':
#     main(sys.argv[1:])
