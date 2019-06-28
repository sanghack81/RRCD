import itertools
import json
import multiprocessing
import sys
import time
from collections import deque
from typing import List

import numpy as np
from pyrcds.domain import RelationalSchema
from pyrcds.model import RCM, \
    generate_values_for_skeleton, enumerate_rdeps, linear_gaussians_rcm
from pyrcds.tests.testing_utils import company_rcm, company_schema
from tqdm import trange

from rrpcd.algorithm import RCMLearner
from rrpcd.data import DataCenter
from rrpcd.experiments.exp_utils import evaluation_for_orientation
from rrpcd.experiments.unified_evaluation import sized_random_skeleton, sizing_method, retrieve_finished
from rrpcd.rci_test import RCITester
from rrpcd.rel_kernel import RBFKernelComputer
from rrpcd.utils import average_aggregator


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


def main(argv):
    tester = None
    KEY_LENGTH = {1: 4, 2: 7}

    is_aggregateds = [True, False]
    sepset_rules = ['minimal']
    orientation_rules = ['majority']
    detect_rbos = [True, False]
    detect_post_rbos = [True, False]

    is_random = 'random' in argv
    is_company = 'company' in argv

    working_dir = get_working_dir(is_company, is_random)
    done = retrieve_finished(KEY_LENGTH, working_dir)

    from_index, to_index, n_jobs, _ = arg_parse(argv)

    if is_random:
        with open(f'data/random/1000_random_schemas.json', 'r') as f:
            schemas = json.load(f)
        with open(f'data/random/1000_random_rcms.json', 'r') as f:
            rcm_codes = json.load(f)
    else:
        schemas, rcm_codes = None, None

    identifier = str(int(time.time() * 100))

    options = list(itertools.product(is_aggregateds, sepset_rules, orientation_rules, detect_rbos, detect_post_rbos))

    p1_queue = deque()
    p2_queue = deque()

    def writing_phase(_phase):
        assert 1 == _phase or 2 == _phase
        queue = p1_queue if _phase == 1 else p2_queue
        with open(f'{working_dir}phase_{_phase}_{from_index}_{to_index}_{identifier}.csv', 'a') as _f:
            while queue:
                vals = queue.popleft()
                print(*vals, file=_f, sep=',')

    last_wrote2 = 0
    for idx in trange(from_index, to_index, smoothing=0):
        for base_size in [200, 300, 400, 500]:  # 200, 300, 400,500, 600
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
                kerner = RBFKernelComputer(datasource, additive=1e-2, n_jobs=n_jobs, eqsize_only=False,
                                           k_cache_max_size=128)
                _tester = RCITester(kerner, n_jobs=n_jobs)
                return _tester

            initialized = False
            for is_aggregated, sepset_rule, orientation_rule, detect_rbo, detect_post_rbo in options:
                if detect_rbo != detect_post_rbo:
                    continue
                if is_aggregated:
                    if not (detect_post_rbo and detect_rbo):
                        continue

                p2_key = (idx, base_size, is_aggregated, sepset_rule, orientation_rule, detect_rbo, detect_post_rbo)

                if p2_key in done[2]:
                    continue

                if not initialized:
                    tester = initialize()
                    initialized = True

                if p2_key not in done[2]:
                    done[2].add(p2_key)
                    np.random.seed(idx + 1)
                    learner = RCMLearner(tester, max_rv_hops=rcm.max_hop, max_degree=None, verbose=False, true_rcm=rcm,
                                         sepset_rule=sepset_rule,
                                         orientation_rule=orientation_rule,
                                         aggregator=average_aggregator if is_aggregated else None,
                                         minimum_rows_for_test=0,
                                         detect_rbo_violations=detect_rbo,
                                         detect_post_rbo_violations=detect_post_rbo)

                    learner.perfect_phase_I()
                    learner.RBO_based_tests()
                    learner.post_RBO_unshielded_triples_tests()
                    learner.orient()

                    p2_values = []
                    p2_values.extend(p2_key)
                    p2_values.append('|')
                    p2_values.extend(learner.rbo_stats[k] for k in stats_keys())
                    p2_values.append('|')
                    p2_values.extend(learner.post_rbo_stats[k] for k in stats_keys())
                    p2_values.append('|')
                    p2_values.extend(evaluation_for_orientation(learner.prcm, rcm)[-6:-3])

                    p2_queue.append(p2_values)

                    if last_wrote2 + 120 < time.time():
                        writing_phase(2)
                        last_wrote2 = time.time()

            if last_wrote2 + 120 < time.time():
                writing_phase(2)
                last_wrote2 = time.time()

    # clean up
    if p1_queue:
        writing_phase(1)
    if p2_queue:
        writing_phase(2)


def arg_parse(argv):
    from_index = int(argv[0])
    to_index = int(argv[1])
    assert 0 <= from_index < to_index <= 300
    n_jobs = int(argv[2]) if len(argv) >= 3 else multiprocessing.cpu_count() // 8
    n_jobs = min(multiprocessing.cpu_count(), max(1, n_jobs))
    experimental = 'experimental' in argv
    print(from_index, to_index, n_jobs, experimental)
    return from_index, to_index, n_jobs, experimental


def get_working_dir(is_company, is_random):
    if is_random and is_company:
        raise ValueError('cannot be random & company')
    if not is_random and not is_company:
        raise ValueError('random or company not specified')
    test_str = 'random' if is_random else 'company'
    working_dir = f'rrpcd/_UAI_2019/{test_str}/'
    return working_dir


if __name__ == '__main__':
    # pyhton -m rrpcd._UAI_2019.random.unified_evaluation_phase_II 0 300 {n_jobs} random
    main(sys.argv[1:])
