import pytest
from pyrcds.domain import AttributeClass

from rrpcd.orienter import OrientationInformation


@pytest.mark.parametrize('rule,threshold,votes,expected', [
    ('percentage', 0.75, (3, 1), 'forward'),
    ('percentage', 0.75, (2, 1), None),
    ('percentage', 0.75, (1, 3), 'reverse'),
    ('majority', None, (2, 1), 'forward'),
    ('conservative', None, (3, 1), None),
])
def test_orientation_threshold(rule, threshold, votes, expected):
    info = OrientationInformation(rule, threshold)
    x, y = AttributeClass('X'), AttributeClass('Y')
    for is_collider, count in zip((True, False), votes):
        for _ in range(count):
            info.add_record((x, y), is_collider)
    orientations = {'forward': {(x, y)}, 'reverse': {(y, x)}, None: set()}
    assert info.orientations() == orientations[expected]
    assert info.undetermined() == ({frozenset({x, y})} if expected is None else set())


@pytest.mark.parametrize('rule,threshold,message', [
    ('unknown', None, 'unknown'),
    ('percentage', 0.49, 'threshold is smaller'),
])
def test_invalid_orientation_options(rule, threshold, message):
    with pytest.raises(AssertionError, match=message):
        OrientationInformation(rule, threshold)
