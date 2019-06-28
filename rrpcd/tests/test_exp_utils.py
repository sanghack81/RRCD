from pyrcds.domain import generate_schema, RelationalSchema


def test_schema():
    schema = generate_schema()
    as_dic = schema.to_dict()
    print()
    print(repr(schema))
    schema2 = RelationalSchema.from_dict(as_dic)
    print(repr(schema2))
    assert schema == schema2
