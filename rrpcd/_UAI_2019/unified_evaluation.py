import multiprocessing
from typing import List

from pyrcds.model import UndirectedRDep


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


def phase_I_to_write(prcm_before, prcm_after, rcm) -> List:
    true_undirecteds = {UndirectedRDep(d) for d in rcm.directed_dependencies}

    before_true_positives = prcm_before.undirected_dependencies & true_undirecteds
    before_false_positives = prcm_before.undirected_dependencies - true_undirecteds

    after_true_positives = prcm_after.undirected_dependencies & true_undirecteds
    after_false_positives = prcm_after.undirected_dependencies - true_undirecteds

    values = list()
    values.append(len(before_true_positives))
    values.append(len(before_false_positives))
    values.append(len(after_true_positives))
    values.append(len(after_false_positives))
    values.append(len(true_undirecteds))
    return values


#
#
# def main(argv):
#     # p1_key = (idx, base_size, is_aggregated, order_dependent)
#     # p2_key = (idx, base_size, is_aggregated, sepset_rule, orientation_rule, detect_rbo, detect_post_rbo)
#     KEY_LENGTH = {1: 4, 2: 7}
#
#     is_aggregateds = [True, False]
#     order_dependents = [True, False]
#     sepset_rules = ['minimal']
#     orientation_rules = ['majority']
#     detect_rbos = [True, False]
#     detect_post_rbos = [True, False]
#
#     is_random = 'random' in argv
#     is_company = 'company' in argv
#
#     working_dir = get_working_dir(is_company, is_random)
#     done = retrieve_finished(KEY_LENGTH, working_dir)
#
#     if '--merge' in argv:
#         for phase in [1, 2]:
#             to_be_merged = list(files(working_dir, prefix=f'phase_{phase}', suffix='.csv'))
#             if to_be_merged:
#                 print(f'merging: ')
#                 for x in to_be_merged:
#                     print('       ', x)
#                 df = pd.concat([pd.read_csv(f'{working_dir}{fname}', header=None) for fname in to_be_merged])
#                 for fname in to_be_merged:
#                     os.rename(f'{working_dir}{fname}', f'{working_dir}{fname}.bak')
#                 df.to_csv(f'{working_dir}phase_{phase}.csv', header=False, index=False)
#             else:
#                 print('nothing to merge.')
#         return
#
#     from_index, to_index, n_jobs, _ = arg_parse(argv)
#
#     if is_random:
#         with open(f'{working_dir}1000_random_schemas.json', 'r') as f:
#             schemas = json.load(f)
#         with open(f'{working_dir}1000_random_rcms.json', 'r') as f:
#             rcm_codes = json.load(f)
#     else:
#         schemas, rcm_codes = None, None
#
#     identifier = str(int(time.time() * 100))
#
#     options = list(itertools.product(is_aggregateds, order_dependents, sepset_rules, orientation_rules, detect_rbos, detect_post_rbos))
#
#     p1_queue = deque()
#     p2_queue = deque()
#
#     def writing_phase(_phase):
#         assert 1 == _phase or 2 == _phase
#         queue = p1_queue if _phase == 1 else p2_queue
#         with open(f'{working_dir}phase_{_phase}_{from_index}_{to_index}_{identifier}.csv', 'a') as _f:
#             while queue:
#                 vals = queue.popleft()
#                 print(*vals, file=_f, sep=',')
#
#     last_wrote1 = 0
#     last_wrote2 = 0
#     for idx in trange(from_index, to_index, smoothing=0):
#         for base_size in [200, 300, 400, 500]:  # 200, 300, 400,500, 600
#             if is_random:
#                 schema = RelationalSchema.from_dict(schemas[idx])
#                 max_hop, rcm_code = rcm_codes[idx]
#                 rdeps = sorted(list(enumerate_rdeps(schema, max_hop)))
#                 dependencies = {rdeps[at] for at in rcm_code}
#                 rcm = RCM(schema, dependencies)
#             else:
#                 schema = company_schema()
#                 rcm = company_rcm()
#
#             def initialize():
#                 np.random.seed(idx + 1)
#                 skeleton = sized_random_skeleton(schema, sizing_method(base_size, schema), seed=idx + 1)
#                 lg_rcm = linear_gaussians_rcm(rcm, seed=idx + 1)
#                 generate_values_for_skeleton(lg_rcm, skeleton, seed=idx + 1)
#
#                 datasource = DataCenter(skeleton)
#                 kerner = RBFKernelComputer(datasource, additive=1e-2, n_jobs=n_jobs, eqsize_only=False, k_cache_max_size=128)
#                 _tester = RCITester(kerner, n_jobs=n_jobs)
#                 return _tester
#
#             initialized = False
#             for is_aggregated, order_dependent, sepset_rule, orientation_rule, detect_rbo, detect_post_rbo in options:
#                 p1_key = (idx, base_size, is_aggregated, order_dependent)
#                 p2_key = (idx, base_size, is_aggregated, sepset_rule, orientation_rule, detect_rbo, detect_post_rbo)
#
#                 if p1_key in done[1] and p2_key in done[2]:
#                     continue
#
#                 if not initialized:
#                     tester = initialize()
#                     initialized = True
#
#                 if p1_key not in done[1]:
#                     done[1].add(p1_key)
#                     """ Phase I """
#                     np.random.seed(idx + 1)
#                     p1_learner = RCMLearner(tester, max_rv_hops=rcm.max_hop, max_degree=None, verbose=False, true_rcm=rcm,
#                                             aggregator=average_aggregator if is_aggregated else None,
#                                             minimum_rows_for_test=0, recover_from_non_RCMC=True, remove_recovered=True,
#                                             phase_I_order_independence=not order_dependent)
#
#                     p1_learner.phase_I()
#
#                     p1_values = []
#                     p1_values.extend(p1_key)
#                     p1_values.extend(phase_I_to_write(p1_learner.prcm_before_post_phase_I, p1_learner.prcm_after_recover, rcm)[0:2])
#                     p1_values.extend(phase_I_to_write(p1_learner.prcm_after_recover, p1_learner.prcm_after_remove_after_recover, rcm))
#
#                     counts = [0, 0, 0]
#                     for cause, effect in {(cause, effect) for cause, effect, _ in p1_learner.saved_by_aggregated_ci}:
#                         dep = RelationalDependency(cause, effect)
#                         rev_dep = dep.reverse()
#                         if UndirectedRDep(dep) not in p1_learner.prcm.undirected_dependencies:
#                             continue
#                         if dep in rcm.directed_dependencies:
#                             counts[0] += 1
#                         elif rev_dep in rcm.directed_dependencies:
#                             counts[1] += 1
#                         else:
#                             counts[2] += 1
#                     p1_values.append(counts[0])
#                     p1_values.append(counts[1])
#                     p1_values.append(counts[2])
#
#                     p1_queue.append(p1_values)
#
#                     if last_wrote1 + 120 < time.time():
#                         writing_phase(1)
#                         last_wrote1 = time.time()
#
#                 if p2_key not in done[2]:
#                     done[2].add(p2_key)
#                     np.random.seed(idx + 1)
#                     learner = RCMLearner(tester, max_rv_hops=rcm.max_hop, max_degree=None, verbose=False, true_rcm=rcm, sepset_rule=sepset_rule,
#                                          orientation_rule=orientation_rule,
#                                          aggregator=average_aggregator if is_aggregated else None,
#                                          minimum_rows_for_test=0, recover_from_non_RCMC=True, remove_recovered=False,
#                                          detect_rbo_violations=detect_rbo,
#                                          detect_post_rbo_violations=detect_post_rbo)
#
#                     learner.perfect_phase_I()
#                     learner.RBO_based_tests()
#                     learner.post_RBO_unshielded_triples_tests()
#                     learner.orient()
#
#                     p2_values = []
#                     p2_values.extend(p2_key)
#                     p2_values.append('|')
#                     p2_values.extend(learner.rbo_stats[k] for k in stats_keys())
#                     p2_values.append('|')
#                     p2_values.extend(learner.post_rbo_stats[k] for k in stats_keys())
#                     p2_values.append('|')
#                     p2_values.extend(evaluation_for_orientation(learner.prcm, rcm)[-6:-3])
#
#                     p2_queue.append(p2_values)
#
#                     if last_wrote2 + 120 < time.time():
#                         writing_phase(2)
#                         last_wrote2 = time.time()
#
#             if last_wrote2 + 120 < time.time():
#                 writing_phase(2)
#                 last_wrote2 = time.time()
#
#     # clean up
#     if p1_queue:
#         writing_phase(1)
#     if p2_queue:
#         writing_phase(2)


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

# if __name__ == '__main__':
#     main(sys.argv[1:])
