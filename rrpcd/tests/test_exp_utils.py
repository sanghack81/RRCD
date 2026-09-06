import pytest
from pyrcds.domain import Cardinality, EntityClass, RelationshipClass, generate_schema, RelationalSchema

from rrpcd.experiments.unified_evaluation import sized_random_skeleton, sizing_method


def test_schema():
    schema = generate_schema()
    as_dic = schema.to_dict()
    print()
    print(repr(schema))
    schema2 = RelationalSchema.from_dict(as_dic)
    print(repr(schema2))
    assert schema == schema2


@pytest.mark.parametrize('card_a, card_b, entity_count, relationship_count', [
    (Cardinality.one, Cardinality.one, 12, 10),
    (Cardinality.one, Cardinality.many, 10, 10),
    (Cardinality.many, Cardinality.one, 10, 10),
    (Cardinality.many, Cardinality.many, 10, 20),
])
def test_sized_random_skeleton(card_a, card_b, entity_count, relationship_count):
    a, b = EntityClass('A', ()), EntityClass('B', ())
    r = RelationshipClass('R', (), {a: card_a, b: card_b})
    schema = RelationalSchema({a, b}, {r})

    skeleton = sized_random_skeleton(schema, sizing_method(10, schema), seed=0)

    assert len(skeleton.items(a)) == entity_count
    assert len(skeleton.items(b)) == entity_count
    assert len(skeleton.items(r)) == relationship_count
    for relation in skeleton.items(r):
        assert len(skeleton.neighbors(relation, a)) == 1
        assert len(skeleton.neighbors(relation, b)) == 1
    for entity_class in (a, b):
        if not r.is_many(entity_class):
            assert all(len(skeleton.neighbors(entity, r)) <= 1
                       for entity in skeleton.items(entity_class))
