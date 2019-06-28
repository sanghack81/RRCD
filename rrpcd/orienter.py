import sys
import time
from collections import namedtuple
from itertools import combinations, chain
from typing import Set, List, Tuple, FrozenSet, Union, TypeVar, Optional

import numpy as np
from pyrcds.domain import AttributeClass
from pyrcds.graphs import PDAG
from pyrcds.model import SymTriple, \
    RCM
from pyrcds.rcds import completes
from pyrcds.utils import group_by

T = TypeVar('T')


def robust_sound_rules(g: PDAG, non_colliders=()):
    """Orient graph based on non-colliders. Return False if any conflicts occur"""
    if any(g.is_oriented_as(x, y) and g.is_oriented_as(z, y) for x, y, z in tuple(non_colliders)):
        return False

    cnt = 0
    while True:
        cnt += 1
        # print('robust_sound_rules',cnt)
        if cnt > 500:
            print('something wrong?')
            break
        mark = len(g.oriented())
        for non_collider in tuple(non_colliders):
            x, y, z = non_collider

            # R1 X-->Y--Z (shielded, and unshielded)
            if g.is_oriented_as(x, y):
                g.orient(y, z)
            # R1' Z-->Y--X (shielded, and unshielded)
            if g.is_oriented_as(z, y):
                g.orient(y, x)

            # R3 (do not check x--z)
            for w in g.ch(x) & g.ne(y) & g.ch(z):
                g.orient(y, w)

            # R4   z-->w-->x
            if g.pa(x) & g.adj(y) & g.ch(z):
                g.orient(y, x)
            # R4'   x-->w-->z
            if g.pa(z) & g.adj(y) & g.ch(x):
                g.orient(y, z)

            if {x, z} <= g.ne(y):
                if z in g.ch(x):
                    g.orient(y, z)
                if x in g.ch(z):
                    g.orient(y, x)

        # R2
        for y in g:
            # x -> y ->z and x -- z
            for x in g.pa(y):
                for z in g.ch(y) & g.ne(x):
                    g.orient(x, z)

        # extended R2
        for y in g:
            # x -> y ->z and x -- z
            for x in g.an(y):
                for z in g.de(y) & g.ne(x):
                    g.orient(x, z)

        if len(g.oriented()) == mark:
            break
    # TODO role of completes?
    if any(g.is_oriented_as(x, y) and g.is_oriented_as(z, y) for x, y, z in tuple(non_colliders)):
        return False
    return True


def __rule_based_selection(rule: str, count1: int, count2: int, percentage_threshold: Optional[float] = None) -> int:
    assert 0 <= count1 and 0 <= count2

    FIRST, SECOND, NOTHING = 0, 1, 2
    if rule == 'conservative':
        rule, percentage_threshold = 'percentage', 1.0
    elif rule == 'majority':
        rule, percentage_threshold = 'percentage', 0.5

    if rule == 'percentage':
        total_count = count1 + count2
        if count1 == count2:
            return NOTHING
        # no tie
        if count1 / total_count >= percentage_threshold:
            return FIRST
        elif count2 / total_count >= percentage_threshold:
            return SECOND
        else:
            return NOTHING
    else:
        raise ValueError(f'unknown rule: {rule}')


def rule_based_selection(rule: str, percentage_threshold=None, **kwargs) -> Optional[str]:
    if len(kwargs) != 2:
        raise ValueError(f'needs exactly two keyword arguments {kwargs}')

    key1, key2 = list(kwargs.keys())
    choice = __rule_based_selection(rule, kwargs[key1], kwargs[key2], percentage_threshold=percentage_threshold)
    return (key1, key2, None)[choice]


class OrientationInformation:
    OriRecord = namedtuple('Record_', ['triple', 'is_collider'])

    def __init__(self, rule, threshold=None):
        assert rule in {'conservative', 'majority', 'percentage'}, f'unknown {self.rule}'
        assert threshold is None or self.threshold >= 0.5, f'threshold is smaller than 0.5 (> {self.threshold}).'

        self.rule = rule
        self.threshold = threshold

        self.records = list()

        self._orientations = None
        self._non_colliders = None
        self._colliders = None
        self._undetermined = None
        self.refined = False

    def add_record(self, xyz_or_xy: Union[SymTriple, Tuple[AttributeClass, AttributeClass]], is_collider: bool):
        self.refined = False
        if not isinstance(xyz_or_xy, SymTriple):
            X, Y = xyz_or_xy
            if X < Y:  # (X,Y,X) collider and (Y,X,Y) non-collider should be treated as the same.
                xyz_or_xy = SymTriple(X, Y, X)
            else:
                xyz_or_xy = SymTriple(Y, X, Y)
                is_collider = not is_collider
        else:
            x, y, z = xyz_or_xy
            assert x != z
            assert x != y
        self.records.append(self.OriRecord(xyz_or_xy, is_collider))

    def __refine(self):
        self._orientations = set()  # type: Set[Tuple[AttributeClass,AttributeClass]]
        self._non_colliders = set()
        self._colliders = set()
        self._undetermined = set()

        for ABC, ab_records in group_by(self.records, lambda record: record.triple):
            A, B, C = ABC
            collier_support = sum(record.is_collider for record in ab_records)
            noncoll_support = len(ab_records) - collier_support

            choice = rule_based_selection(self.rule,
                                          as_collider=collier_support,
                                          as_noncoll=noncoll_support,
                                          percentage_threshold=self.threshold)
            if choice == 'as_collider':
                if A == C:
                    self._orientations.add((A, B))
                else:
                    self._colliders.add(ABC)
            elif choice == 'as_noncoll':
                if A == C:
                    self._orientations.add((B, A))
                else:
                    self._non_colliders.add(ABC)
            else:
                self._undetermined.add(frozenset({A, B}) if A == C else ABC)

        self._undetermined -= {frozenset(AB) for AB in self._orientations}
        self.refined = True

    def orientations(self) -> Set[Tuple[AttributeClass, AttributeClass]]:
        if not self.refined:
            self.__refine()
        return set(self._orientations)

    def non_colliders(self) -> Set[SymTriple]:
        if not self.refined:
            self.__refine()
        return set(self._non_colliders)

    def colliders(self) -> Set[SymTriple]:
        if not self.refined:
            self.__refine()
        return set(self._colliders)

    def undetermined(self) -> Set[FrozenSet[AttributeClass]]:
        if not self.refined:
            self.__refine()
        return set(self._undetermined)

    def report(self, true_rcm: RCM = None):
        true_cdg = true_rcm.class_dependency_graph if true_rcm else None
        print('---------- Orientation Information -----------')
        attributes = set(chain(*[list(xy_or_xyz) for xy_or_xyz in
                                 self.orientations() | self.undetermined() | self.non_colliders() | self.colliders()]))
        just = max(len(str(attr)) for attr in attributes) + 2

        for X, Y in self.orientations():
            count_top, count_bottom = 0, 0
            if X < Y:
                xyx = SymTriple(X, Y, X)
                for record in self.records:
                    if record.triple == xyx:
                        if record.is_collider:
                            count_top += 1
                        count_bottom += 1
            else:
                yxy = SymTriple(Y, X, Y)
                for record in self.records:
                    if record.triple == yxy:
                        if not record.is_collider:
                            count_top += 1
                        count_bottom += 1
            wrongly_directed = true_rcm is not None and true_cdg.is_oriented_as(Y, X)
            print(f'{str(X).center(just)} --> {str(Y).center(just)}     {"".center(just)} with ({count_top}/{count_bottom}) (RBO)', file=sys.stderr if wrongly_directed else sys.stdout,
                  flush=True)
            if wrongly_directed:
                time.sleep(0.1)
        for xyz in self.colliders():
            X, Y, Z = xyz
            count_top, count_bottom = 0, 0
            for record in self.records:
                if record.triple == xyz:
                    if record.is_collider:
                        count_top += 1
                    count_bottom += 1
            wrongly_directed = true_rcm is not None and (true_cdg.is_oriented_as(Y, X) or true_cdg.is_oriented_as(Y, Z))
            print(f'{str(X).center(just)} --> {str(Y).center(just)} <-- {str(Z).center(just)} with ({count_top}/{count_bottom}) (Collider)',
                  file=sys.stderr if wrongly_directed else sys.stdout,
                  flush=True)
            if wrongly_directed:
                time.sleep(0.1)
        for xyz in self.non_colliders():
            X, Y, Z = xyz
            count_top, count_bottom = 0, 0
            for record in self.records:
                if record.triple == xyz:
                    if not record.is_collider:
                        count_top += 1
                    count_bottom += 1
            wrongly_directed = true_rcm is not None and true_cdg.is_oriented_as(X, Y) and true_cdg.is_oriented_as(Z, Y)
            print(f'{str(X).center(just)} --- {str(Y).center(just)} --- {str(Z).center(just)} with ({count_top}/{count_bottom}) (Non-collider)',
                  file=sys.stderr if wrongly_directed else sys.stdout,
                  flush=True)
            if wrongly_directed:
                time.sleep(0.1)
        for xy_or_xyz in self.undetermined():
            if isinstance(xy_or_xyz, SymTriple):
                xyz = xy_or_xyz
                X, Y, Z = xyz
                count_top, count_bottom = 0, 0
                for record in self.records:
                    if record.triple == xyz:
                        if record.is_collider:
                            count_top += 1
                        else:
                            count_bottom += 1

                print(f'{str(X).center(just)} -?- {str(Y).center(just)} -?- {str(Z).center(just)} with (collider: {count_top}, non-collider: {count_bottom})', flush=True)
            else:
                X, Y = xy_or_xyz
                count_top, count_bottom = 0, 0
                if X < Y:
                    xyx = SymTriple(X, Y, X)
                    for record in self.records:
                        if record.triple == xyx:
                            if record.is_collider:
                                count_top += 1
                            else:
                                count_bottom += 1
                else:
                    yxy = SymTriple(Y, X, Y)
                    for record in self.records:
                        if record.triple == yxy:
                            if not record.is_collider:
                                count_top += 1
                            else:
                                count_bottom += 1
                print(f'{str(X).center(just)} -?- {str(Y).center(just)}     {"".center(just)} with (X-->Y: {count_top}, X<--Y: {count_bottom})', flush=True)
        print('------------------------------------')


class Orienter:
    def __init__(self, pdag: PDAG, orientation_info: OrientationInformation, background_knowledge=frozenset()):
        singles, colliders, non_colliders = orientation_info.orientations(), orientation_info.colliders(), orientation_info.non_colliders()

        self.pdag = pdag.copy()

        # TODO validate background_knowledge
        for x, y in background_knowledge:  # can be considered as ancestral relationship
            if not self.pdag.is_adj(x, y):
                self.pdag.add_edge(x, y)

        assert not any((y, x) in singles for x, y in singles)
        assert not (colliders & non_colliders)
        # singles = {(x, y) for x, y in singles if (y, x) not in singles}
        # colliders, non_colliders = colliders - non_colliders, non_colliders - colliders

        if background_knowledge:
            singles = {(x, y) for x, y in singles if
                       (x, y) not in background_knowledge and (y, x) not in background_knowledge}
            colliders = {SymTriple(x, y, z) for x, y, z in colliders if
                         (y, x) not in background_knowledge and (y, z) not in background_knowledge}
            non_colliders = {SymTriple(x, y, z) for x, y, z in non_colliders if
                             (y, x) in background_knowledge or (y, z) in background_knowledge}

        self.singles = set(singles)
        self.colliders = set(colliders)
        self.non_colliders = set(non_colliders)
        self.undetermined = set(orientation_info.undetermined())

    def sequential_orient(self, strategy='mostly-shared') -> PDAG:
        # maximize orientation
        # TODO efficiency
        # all_info = list(self.singles | self.colliders | self.non_colliders)
        all_info = sorted(self.singles) + sorted(self.colliders) + sorted(self.non_colliders)
        all_info = list(all_info)
        np.random.shuffle(all_info)
        gs = list()  # type: List[PDAG]
        # print(len(all_info))

        sub_infos = list()
        for current_info in all_info:
            sub_info = set(sub_infos + [current_info])
            # quick validation
            violated = False
            for xyz in sub_info:
                if isinstance(xyz, SymTriple) and xyz in self.colliders:
                    X, Y, Z = xyz
                    if (Y, X) in sub_info or (Y, Z) in sub_info:
                        violated = True
                        break
            if violated:
                continue
            # TODO two colliders conflict

            insta_non_collider = set()
            for xyz in self.undetermined:
                if isinstance(xyz, SymTriple):
                    assert xyz not in sub_info
                    X, Y, Z = xyz
                    # X<--Y--Z
                    # X--Y-->Z
                    if (Y, X) in sub_info or (Y, Z) in sub_info:
                        insta_non_collider.add(xyz)

            outcome = self.__orient_with(sub_info | insta_non_collider)
            if outcome is not None:
                # gs.append(outcome)
                gs = outcome
                sub_infos.append(current_info)

        if gs is not None:
            return gs
        raise NotImplementedError()

    def orient(self, strategy='mostly-shared') -> PDAG:
        # maximize orientation
        # TODO efficiency
        # all_info = list(self.singles | self.colliders | self.non_colliders)
        all_info = sorted(self.singles) + sorted(self.colliders) + sorted(self.non_colliders)
        gs = list()  # type: List[PDAG]
        # print(len(all_info))
        for num_remove in range(len(all_info) + 1):
            for sub_info in combinations(all_info, len(all_info) - num_remove):
                sub_info = set(sub_info)
                # quick validation
                violated = False
                for xyz in sub_info:
                    if isinstance(xyz, SymTriple) and xyz in self.colliders:
                        X, Y, Z = xyz
                        if (Y, X) in sub_info or (Y, Z) in sub_info:
                            violated = True
                            break
                if violated:
                    continue
                # TODO two colliders conflict

                insta_non_collider = set()
                for xyz in self.undetermined:
                    if isinstance(xyz, SymTriple):
                        assert xyz not in sub_info
                        X, Y, Z = xyz
                        # X<--Y--Z
                        # X--Y-->Z
                        if (Y, X) in sub_info or (Y, Z) in sub_info:
                            insta_non_collider.add(xyz)

                outcome = self.__orient_with(sub_info | insta_non_collider)
                if outcome is not None:
                    gs.append(outcome)
            if gs:
                break
        assert gs

        if len(gs) == 1:
            return gs[0]
        if strategy == 'mostly-shared':
            if len(gs) >= 3:
                g_score = [0] * len(gs)
                for i, g in enumerate(gs):
                    g_score[i] = sum(len(other_g.oriented() & g.oriented())
                                     for other_g in gs[:i] + gs[i + 1:])
                max_at = np.argmax(g_score)  # type: int
                return gs[max_at]
            else:  # len == 2
                return gs[0] if len(gs[0].oriented()) >= len(gs[1].oriented()) else gs[1]
        else:
            raise NotImplementedError()

    def __orient_with(self, sub_info) -> Optional[PDAG]:
        working_pdag = self.pdag.copy()

        def orient(_x, _y) -> bool:
            assert working_pdag.is_adj(_x, _y)
            if working_pdag.is_oriented_as(_x, _y):
                return True
            elif working_pdag.is_unoriented(_x, _y):
                working_pdag.orient(_x, _y)
                return True
            if working_pdag.is_oriented_as(_y, _x):
                return False

        non_colliders = set()
        for info in sub_info:
            if isinstance(info, SymTriple):
                if info in self.colliders:
                    x, y, z = info
                    if not orient(x, y) or not orient(z, y):
                        return None
                else:
                    non_colliders.add(info)
            else:
                x, y = info
                if not orient(x, y):
                    return None

        if not robust_sound_rules(working_pdag, non_colliders):
            return None
        completes(working_pdag, non_colliders)
        return working_pdag
