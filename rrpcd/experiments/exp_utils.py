import os

import numpy as np
import pandas as pd
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
