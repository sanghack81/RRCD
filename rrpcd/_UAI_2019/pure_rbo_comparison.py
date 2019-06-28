import json
import multiprocessing
import time
from collections import deque

import numpy as np
from pyrcds.domain import RelationalSchema
from pyrcds.model import RCM, \
    generate_values_for_skeleton, enumerate_rdeps, linear_gaussians_rcm
from pyrcds.tests.testing_utils import company_rcm, company_schema
from tqdm import trange

from rrpcd.algorithm import RCMLearner
from rrpcd.data import DataCenter
from rrpcd.experiments.unified_evaluation import sized_random_skeleton, sizing_method
from rrpcd.rci_test import RCITester
from rrpcd.rel_kernel import RBFKernelComputer


def examine_oriori(learner, rcm):
    learner.orientation_info.records
    CDG = rcm.class_dependency_graph
    category_cnts = [0, 0, 0, 0]
    for record in learner.orientation_info.records:
        x, y, z = record.triple
        if record.is_collider:
            if CDG.is_oriented_as(x, y) and CDG.is_oriented_as(z, y):
                category_cnts[0] += 1  # 'correct collider'
            else:
                category_cnts[1] += 1  # 'wrong collider'
        else:
            if CDG.is_oriented_as(x, y) and CDG.is_oriented_as(z, y):
                category_cnts[2] += 1  # 'wrong non-collider'
            else:
                category_cnts[3] += 1  # 'correct non-collider'
    return category_cnts


def main(argv):
    tester = None

    sepset_rule = 'first'
    orientation_rule = 'majority'

    is_random = 'random' in argv
    is_company = 'company' in argv

    working_dir = get_working_dir(is_company, is_random)

    from_index, to_index, n_jobs, _ = arg_parse(argv)

    if is_random:
        with open(f'data/random/1000_random_schemas.json', 'r') as f:
            schemas = json.load(f)
        with open(f'data/random/1000_random_rcms.json', 'r') as f:
            rcm_codes = json.load(f)
    else:
        schemas, rcm_codes = None, None

    identifier = str(int(time.time() * 100))

    p1_queue = deque()
    p2_queue = deque()

    def writing_phase(_phase):
        assert _phase == 2
        assert 1 == _phase or 2 == _phase
        queue = p1_queue if _phase == 1 else p2_queue
        with open(f'{working_dir}pure_rbo_comparison_{_phase}_{from_index}_{to_index}_{identifier}.csv', 'a') as _f:
            while queue:
                vals = queue.popleft()
                print(*vals, file=_f, sep=',')

    last_wrote2 = 0
    for idx in trange(from_index, to_index, smoothing=0):
        for base_size in [200, 500]:
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

            if not initialized:
                tester = initialize()
                initialized = True

            np.random.seed(idx + 1)
            learner0 = RCMLearner(tester, max_rv_hops=rcm.max_hop, max_degree=None, verbose=False, true_rcm=rcm, sepset_rule=sepset_rule,
                                  orientation_rule=orientation_rule,
                                  aggregator=None,
                                  minimum_rows_for_test=0,
                                  detect_rbo_violations=False,
                                  detect_post_rbo_violations=False)

            learner1 = RCMLearner(tester, max_rv_hops=rcm.max_hop, max_degree=None, verbose=False, true_rcm=rcm, sepset_rule=sepset_rule,
                                  orientation_rule=orientation_rule,
                                  aggregator=None,
                                  minimum_rows_for_test=0,
                                  detect_rbo_violations=False,
                                  detect_post_rbo_violations=False)

            learner2 = RCMLearner(tester, max_rv_hops=rcm.max_hop, max_degree=None, verbose=False, true_rcm=rcm, sepset_rule=sepset_rule,
                                  orientation_rule=orientation_rule,
                                  aggregator=None,
                                  minimum_rows_for_test=0,
                                  detect_rbo_violations=True,
                                  detect_post_rbo_violations=False)

            learner0.perfect_phase_I()
            learner0.CUT_based_collider_tests(rbo_only=True)

            learner1.perfect_phase_I()
            learner1.RBO_based_tests()

            learner2.perfect_phase_I()
            learner2.RBO_based_tests()

            p2_values = []
            p2_values.extend((idx, base_size))
            p2_values.extend(examine_oriori(learner0, rcm))
            p2_values.extend(examine_oriori(learner1, rcm))
            p2_values.extend(examine_oriori(learner2, rcm))
            print(p2_values)
            # exit(0)
            p2_queue.append(p2_values)

            if last_wrote2 + 120 < time.time():
                writing_phase(2)
                last_wrote2 = time.time()

    # clean up
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


def analyze(working_dir):
    import pandas as pd
    df = pd.read_csv(f'{working_dir}pure_rbo_comparison.csv', header=None)
    df.columns = ('idx', 'base size',
                  'CC0', 'WC0', 'WNC0', 'CNC0',
                  'CC1', 'WC1', 'WNC1', 'CNC1',
                  'CC2', 'WC2', 'WNC2', 'CNC2')

    for size in [200, 500]:
        print()
        print(size)
        fixed_df = df[df['base size'] == size].sum()
        for i in [0, 1, 2]:
            accuracy = (fixed_df[f'CC{i}'] + fixed_df[f'CNC{i}']) / (fixed_df[f'CC{i}'] + fixed_df[f'CNC{i}'] + fixed_df[f'WC{i}'] + fixed_df[f'WNC{i}'])
            actual_colliders = (fixed_df[f'CC{i}'] + fixed_df[f'WNC{i}'])
            actual_noncolliders = (fixed_df[f'CNC{i}'] + fixed_df[f'WC{i}'])
            collider_accuracy = fixed_df[f'CC{i}'] / actual_colliders
            noncollider_accuracy = fixed_df[f'CNC{i}'] / actual_noncolliders
            print(i, f'{accuracy * 100:.1f}', f'{collider_accuracy * 100:.1f}', f'{noncollider_accuracy * 100:.1f}')


#

if __name__ == '__main__':
    # main(sys.argv[1:])
    analyze(get_working_dir(False, True))
    if False:
        main(["0", "300", "4", "random"])
