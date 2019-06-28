import itertools
import json
import multiprocessing
import os
import sys
import time
from collections import deque
from typing import List

import numpy as np
import pandas as pd
from pyrcds.domain import RelationalSchema
from pyrcds.model import RCM, \
    generate_values_for_skeleton, enumerate_rdeps, linear_gaussians_rcm, RelationalDependency
from pyrcds.model import UndirectedRDep
from pyrcds.tests.testing_utils import company_rcm, company_schema
from tqdm import trange

from rrpcd.algorithm import RCMLearner
from rrpcd.data import DataCenter
from rrpcd.experiments.exp_utils import files
from rrpcd.experiments.unified_evaluation import sizing_method, sized_random_skeleton, retrieve_finished
from rrpcd.rci_test import RCITester
from rrpcd.rel_kernel import RBFKernelComputer
from rrpcd.utils import average_aggregator, mkdirs


def phase_I_to_write(prcm, rcm) -> List:
    true_undirecteds = {UndirectedRDep(d) for d in rcm.directed_dependencies}

    TP = prcm.undirected_dependencies & true_undirecteds
    FP = prcm.undirected_dependencies - true_undirecteds

    return [len(TP), len(FP), len(true_undirecteds)]


def main(argv):
    # p1_key = (idx, base_size, is_aggregated, order_dependent)
    KEY_LENGTH = {1: 4, 2: 7}

    is_aggregateds = [True, False]
    order_dependents = [True, False]

    is_random = 'random' in argv
    is_company = 'company' in argv

    working_dir = get_working_dir(is_company, is_random)
    done = retrieve_finished(KEY_LENGTH, working_dir)

    if '--merge' in argv:
        for phase in [1, 2]:
            to_be_merged = list(files(working_dir, prefix=f'phase_{phase}', suffix='.csv'))
            if to_be_merged:
                print(f'merging: ')
                for x in to_be_merged:
                    print('       ', x)
                df = pd.concat([pd.read_csv(f'{working_dir}{fname}', header=None) for fname in to_be_merged])
                for fname in to_be_merged:
                    os.rename(f'{working_dir}{fname}', f'{working_dir}{fname}.bak')
                df.to_csv(f'{working_dir}phase_{phase}.csv', header=False, index=False)
            else:
                print('nothing to merge.')
        return

    from_index, to_index, n_jobs = arg_parse(argv)

    if is_random:
        with open(f'data/random/1000_random_schemas.json', 'r') as f:
            schemas = json.load(f)
        with open(f'data/random/1000_random_rcms.json', 'r') as f:
            rcm_codes = json.load(f)
    else:
        schemas, rcm_codes = None, None

    identifier = str(int(time.time() * 100))

    options = list(itertools.product(is_aggregateds, order_dependents))

    p1_queue = deque()
    tester = None

    def writing_phase(_phase):
        assert 1 == _phase
        queue = p1_queue
        with open(f'{working_dir}phase_{_phase}_{from_index}_{to_index}_{identifier}.csv', 'a') as _f:
            while queue:
                vals = queue.popleft()
                print(*vals, file=_f, sep=',')

    last_wrote1 = 0
    for idx in trange(from_index, to_index, smoothing=0):
        for base_size in [200, 300, 400, 500]:
            if is_random:
                schema = RelationalSchema.from_dict(schemas[idx])
                max_hop, rcm_code = rcm_codes[idx]
                rdeps = sorted(list(enumerate_rdeps(schema, max_hop)))
                dependencies = {rdeps[at] for at in rcm_code}
                rcm = RCM(schema, dependencies)
            else:
                schema = company_schema()
                rcm = company_rcm()

            def initialize():
                np.random.seed(idx + 1)
                skeleton = sized_random_skeleton(schema, sizing_method(base_size, schema), seed=idx + 1)
                lg_rcm = linear_gaussians_rcm(rcm, seed=idx + 1)
                generate_values_for_skeleton(lg_rcm, skeleton, seed=idx + 1)

                datasource = DataCenter(skeleton)
                kerner = RBFKernelComputer(datasource, additive=1e-2, n_jobs=n_jobs, eqsize_only=False, k_cache_max_size=128)
                _tester = RCITester(kerner, n_jobs=n_jobs)
                return _tester

            initialized = False
            for is_aggregated, order_dependent in options:
                p1_key = (idx, base_size, is_aggregated, order_dependent)

                if p1_key in done[1]:
                    continue

                if not initialized:
                    tester = initialize()
                    initialized = True

                if p1_key not in done[1]:
                    done[1].add(p1_key)
                    """ Phase I """
                    np.random.seed(idx + 1)
                    p1_learner = RCMLearner(tester, max_rv_hops=rcm.max_hop, max_degree=None, verbose=False, true_rcm=rcm,
                                            aggregator=average_aggregator if is_aggregated else None,
                                            minimum_rows_for_test=0,
                                            phase_I_order_independence=not order_dependent)

                    p1_learner.phase_I()

                    p1_values = []
                    p1_values.extend(p1_key)
                    p1_values.extend(phase_I_to_write(p1_learner.prcm, rcm))

                    counts = [0, 0, 0]
                    for cause, effect in {(cause, effect) for cause, effect, _ in p1_learner.saved_by_aggregated_ci}:
                        dep = RelationalDependency(cause, effect)
                        rev_dep = dep.reverse()
                        if UndirectedRDep(dep) not in p1_learner.prcm.undirected_dependencies:
                            continue
                        if dep in rcm.directed_dependencies:
                            counts[0] += 1
                        elif rev_dep in rcm.directed_dependencies:
                            counts[1] += 1
                        else:
                            counts[2] += 1
                    p1_values.append(counts[0])
                    p1_values.append(counts[1])
                    p1_values.append(counts[2])

                    p1_queue.append(p1_values)

                    if last_wrote1 + 120 < time.time():
                        writing_phase(1)
                        last_wrote1 = time.time()

    # clean up
    if p1_queue:
        writing_phase(1)


def arg_parse(argv):
    from_index = int(argv[0])
    to_index = int(argv[1])
    assert 0 <= from_index < to_index <= 300
    n_jobs = int(argv[2]) if len(argv) >= 3 else multiprocessing.cpu_count() // 2
    n_jobs = min(multiprocessing.cpu_count(), max(1, n_jobs))

    print(from_index, to_index, n_jobs)
    return from_index, to_index, n_jobs


def get_working_dir(is_company, is_random):
    if is_random and is_company:
        raise ValueError('cannot be random & company')
    if not is_random and not is_company:
        raise ValueError('random or company not specified')
    test_str = 'random' if is_random else 'company'
    working_dir = f'rrpcd/_UAI_2019/{test_str}/'
    mkdirs(working_dir)
    return working_dir


if __name__ == '__main__':
    # from, to, n_jobs
    main(sys.argv[1:])
