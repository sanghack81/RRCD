# import itertools
#
# import json
# import networkx as nx
import os

import numpy as np
import pandas as pd
# from pyrcds.domain import RelationalSchema, EntityClass, Cardinality, RelationshipClass, RelationalSkeleton, SkItem
from pyrcds.model import UndirectedRDep, PRCM, RCM
from pyrcds.rcds import markov_equivalence


def evaluation_for_orientation(prcm: PRCM, rcm: RCM):
    mprcm = markov_equivalence(rcm)
    mpcdg = mprcm.class_dependency_graph

    pcdg = prcm.class_dependency_graph
    cdg = rcm.class_dependency_graph

    true_unds_cdg = {frozenset((x, y)) for x, y in cdg.oriented()}
    pred_unds_cdg = {frozenset((x, y)) for x, y in pcdg.oriented()} | pcdg.unoriented()

    true_und_deps = {UndirectedRDep(d) for d in rcm.directed_dependencies}
    pred_und_deps = prcm.undirected_dependencies | {UndirectedRDep(d) for d in prcm.directed_dependencies}

    num_correct_und_deps = len(true_und_deps & pred_und_deps)
    num_correct_und_cdg = len(true_unds_cdg & pred_unds_cdg)
    num_correct_dir_deps = len(prcm.directed_dependencies & rcm.directed_dependencies)
    num_correct_dir_cdg = len(pcdg.oriented() & cdg.oriented())
    num_correct_dir_deps_me = len(prcm.directed_dependencies & mprcm.directed_dependencies)
    num_correct_dir_cdg_me = len(pcdg.oriented() & mpcdg.oriented())

    return (num_correct_und_deps, len(pred_und_deps), len(true_und_deps),
            num_correct_und_cdg, len(pred_unds_cdg), len(true_unds_cdg),
            num_correct_dir_deps, len(prcm.directed_dependencies), len(rcm.directed_dependencies),
            num_correct_dir_cdg, len(pcdg.oriented()), len(cdg.oriented()),
            num_correct_dir_deps_me, len(prcm.directed_dependencies), len(mprcm.directed_dependencies),
            num_correct_dir_cdg_me, len(pcdg.oriented()), len(mpcdg.oriented()))


def fixed(df: pd.DataFrame, fixers: dict, not_equal=False) -> pd.DataFrame:
    """ DataFrame with fixed values as defined in fixers and not_equal option """
    selector = None
    for k, v in fixers.items():
        if selector is None:
            if not_equal:
                selector = (df[k] != v)
            else:
                selector = (df[k] == v)
        else:
            if not_equal:
                selector = np.logical_and(selector, df[k] != v)
            else:
                selector = np.logical_and(selector, df[k] == v)
    if selector is None:
        return df.copy()
    else:
        return df[selector].reset_index(drop=True).copy()


def files(path: str, prefix: str = None, suffix: str = None):
    """ List files with given prefix and suffix """
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if (prefix is None or file.startswith(prefix)) and (suffix is None or file.endswith(suffix)):
                yield file

# def skeleton_to_json(skeleton: RelationalSkeleton):
#     as_dict = dict()
#     as_dict['schema'] = skeleton.schema.to_dict()
#     as_dict['entities'] = {ec: [{'name': ec_item.name, 'values': {attr: ec_item[attr] for attr in ec.attrs}} for ec_item in skeleton.items(ec)] for ec in skeleton.schema.entity_classes}
#     as_dict['relationships'] = {rc: [{'name': rc_item.name, 'values': {attr: rc_item[attr] for attr in rc.attrs}} for rc_item in skeleton.items(rc)] for rc in skeleton.schema.relationship_classes}
#     as_dict['edges'] = {r.name: [e.name for e in skeleton.neighbors(r)] for rc in skeleton.schema.relationship_classes for r in skeleton.items(rc)}
#
#     return json.dumps(as_dict)


# def json_to_skeleton(json_string):
#     # TODO inefficient due to validating cardinality, etc.
#     as_dict = json.loads(json_string)
#     skeleton = RelationalSkeleton(RelationalSchema.from_dict(as_dict['schema']))
#     name2entity = dict()
#     for ec, ec_item_infos in as_dict['entities'].items():
#         for ec_item_info in ec_item_infos:
#             entity = SkItem(ec_item_info['name'], ec, ec_item_info['values'])
#             skeleton.add_entity(entity)
#             name2entity[entity.name] = entity
#
#     for rc, rc_item_infos in as_dict['relationships'].items():
#         for rc_item_info in rc_item_infos:
#             relationship = SkItem(rc_item_info['name'], rc, rc_item_info['values'])
#             neighbors = [name2entity[entity_name] for entity_name in as_dict['edges'][relationship.name]]
#             skeleton.add_relationship(relationship, neighbors)
#     return skeleton


# def enumerate_schemas_without_attrs(num_ent_classes, max_rel_classes=None, arity_range_inclusive=(2, 3)) -> RelationalSchema:
#     """A random relational schema.
#
#     Notes
#     -----
#     Guarantees reproducibility
#     """
#     ent_classes = []
#     rel_classes = []
#
#     assert 2 <= min(arity_range_inclusive)
#     assert 1 <= num_ent_classes
#
#     for i in range(1, num_ent_classes + 1):
#         ent_classes.append(EntityClass("E" + str(i), []))
#
#     i = 1
#     for arity in range(min(arity_range_inclusive), max(arity_range_inclusive) + 1):
#         for selected_es in itertools.combinations(ent_classes, arity):
#             cards = {e: Cardinality.many for e in selected_es}
#             rel_classes.append(RelationshipClass("R" + str(i), [], cards))
#             i += 1
#
#     if max_rel_classes is not None and len(rel_classes) > max_rel_classes:
#         bases = list(itertools.combinations(rel_classes, max_rel_classes))
#     else:
#         bases = [rel_classes]
#
#     while bases:
#         newbases = set()
#         for base in list(bases):
#             schema = RelationalSchema(ent_classes, base)
#             if schema_connected(schema):
#                 yield schema
#             else:
#                 continue
#             newbases |= set(itertools.combinations(base, len(base) - 1))
#         bases = newbases
#
#     return RelationalSchema(ent_classes, rel_classes)


# def to_ug(schema: RelationalSchema) -> nx.Graph:
#     """An undirected graph representation (networkx.Graph) of the relational skeleton."""
#     g = nx.Graph()
#     g.add_nodes_from(schema.entities, kind='E')
#     g.add_nodes_from(schema.relationships, kind='R')
#     for r in schema.relationships:
#         for e in r.entities:
#             g.add_edge(e, r, kind=r.is_many(e))
#
#     g.add_nodes_from(schema.attrs, kind='A')
#     for attr in schema.attrs:
#         g.add_edge(schema.item_class_of(attr), attr)
#     return g


# def schema_equals(s1: RelationalSchema, s2: RelationalSchema) -> bool:
#     return nx.is_isomorphic(to_ug(s1), to_ug(s2), node_match=lambda x, y: x == y, edge_match=lambda x, y: x == y)


# def schema_connected(s1: RelationalSchema) -> bool:
#     ug = s1.as_networkx_ug()  # type: nx.Graph
#     return nx.is_connected(ug)


# def duplicate_relationship(s1: RelationalSchema, with_cardinality=False):
#     for r1, r2 in itertools.combinations(sorted(s1.relationship_classes), 2):
#         if r1.entity_classes == r2.entity_classes:
#             if with_cardinality:
#                 profile1 = [r1.is_many(e) for e in sorted(r1.entity_classes)]
#                 profile2 = [r2.is_many(e) for e in sorted(r1.entity_classes)]
#                 if profile1 == profile2:
#                     return True
#             else:
#                 return True
#     return False
