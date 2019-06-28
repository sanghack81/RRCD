import os
from contextlib import contextmanager
from itertools import combinations
from typing import Iterable, List, TypeVar, Set, Optional, Collection

import numpy as np
from pyrcds.model import RelationalPath

T = TypeVar('T')


def mkdirs(newdir, mode=0o777):
    os.makedirs(newdir, mode=mode, exist_ok=True)


def average_aggregator(vals: Collection[float]) -> float:
    """ Average of values only when the given values are not empty """
    if len(vals) == 0:
        raise ValueError('no empty values')
    return sum(vals) / len(vals)


def with_default(v: Optional[T], dflt: Optional[T]) -> Optional[T]:
    return dflt if v is None else v


def set_combinations(s: Iterable[T], n: int, randomized=False) -> Iterable[Set[T]]:
    if randomized:
        sets = [set(c) for c in combinations(sorted(s), n)]
        np.random.shuffle(sets)
        return sets
    else:
        return (set(c) for c in combinations(sorted(s), n))


def is_1to1(P: RelationalPath) -> bool:
    """ Whether a relational path is one to one relationship between the starting and ending item classes."""
    return not P.is_many and not P.reverse().is_many


def refine_with(selector, *args):
    for arg in args:
        yield [arg[idx] for idx in selector]


def mul2(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return x * y


def shuffled(container) -> List:
    copied = sorted(list(container))
    np.random.shuffle(copied)
    return copied


def pick(vals: Iterable[T]) -> T:
    vals = list(vals)
    if len(vals) == 1:
        return vals[0]
    else:
        return vals[np.random.randint(len(vals))]


def multiplys(*args):
    """Multiplying all matrices, None for empty"""
    temp = None
    for arg in args:
        if arg is not None:
            if temp is None:
                temp = arg.copy()
            else:
                temp *= arg
    return temp


def reproducible(func):
    """ Wrap a function to add `seed' for reducible research """

    def rep_func(*args, seed=None, **kwargs):
        with seeded(seed):
            return func(*args, **kwargs)

    return rep_func


@contextmanager
def seeded(seed=None):
    """ Provides a context to control randomness with given seeds """
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
        yield
        np.random.set_state(st0)
    else:
        yield
