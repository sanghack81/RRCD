# import sys
# import time
# import warnings
# from collections import defaultdict
# from itertools import product, combinations
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import scipy
# import seaborn as sns
# from scipy.stats import ttest_ind
#
# from rrpcd.experiments.exp_utils import fixed
# from rrpcd.experiments.unified_evaluation import stats_keys
# from rrpcd.utils import reproducible
#
# warnings.filterwarnings("error")
# # Pycharm debugger uses where script exists ...
# # working_dir = f''
# working_dir = f'rrpcd/experiments/'
#
# phase_1_id_vars = ('idx', 'base size', 'aggregation', 'order-dependence')
# phase_1_columns = (
#     *phase_1_id_vars,
#     'initial true positives', 'initial false positives',
#     'recovered true positives', 'recovered false positives',
#     'refined true positives', 'refined false positives',
#     'dependencies',
#     'right direction (aggregated)', 'reverse direction (aggregated)', 'false positives (aggregated)'
# )
#
# phase_2_id_vars = ('idx', 'base size', 'aggregation', 'sepset rule', 'orientation rule', 'detect rbo', 'detect post rbo')
# phase_2_columns = (*phase_2_id_vars, 'dummy0',
#                    *['RBO ' + k for k in stats_keys()], 'dummy1',
#                    *['post-RBO ' + k for k in stats_keys()], 'dummy2',
#                    'correct directed', 'directed', 'directables'
#                    )
#
#
# def inspect_fixed(master_df, df=None):
#     if df is None:
#         colval = dict()
#         for col in master_df.columns:
#             if len(master_df[col].unique()) == 1:
#                 colval[col] = master_df[col].unique()[0]
#         if colval:
#             print(f">> given data frame with a single value: {colval}", file=sys.stderr, flush=True)
#     else:
#         removed = set(master_df.columns) - set(df.columns)
#         new_cols = set(df.columns) - set(master_df.columns)
#         if removed:
#             print(f'>> removed columns: {removed}')
#         if new_cols:
#             print(f'>> new columns:     {[col for col in df.columns if col in new_cols]}')
#         for col in df.columns:
#             if len(df[col].unique()) == 1:
#                 print(f">> used data frame with a single value: {col.rjust(30)}: {df[col].unique()[0]}", file=sys.stderr, flush=True)
#
#
# def logged(f):
#     def fff(label: str, master_df: pd.DataFrame, *args, **kwargs):
#         time.sleep(0.1)
#         print("\n\n\n\n", file=sys.stderr, flush=True)
#         print("===================================================", file=sys.stderr, flush=True)
#         print(f'starting {f.__name__} for label: {label}', file=sys.stderr, flush=True)
#         inspect_fixed(master_df)
#         time.sleep(0.1)
#         df = f(label, master_df, *args, **kwargs)
#         time.sleep(0.1)
#         if df is not None:
#             inspect_fixed(master_df, df)
#
#         print('ending   ' + f.__name__, file=sys.stderr, flush=True)
#         print("==================================================", file=sys.stderr, flush=True)
#         time.sleep(0.1)
#
#     return fff
#
#
# def _print_(*args, verbose=False, **kwargs):
#     if verbose:
#         print(*args, **kwargs)
#
#
# def _find_worse_settings(master_df, criteria, columns, alpha=0.01, verbose=False, worst=False, the_larger_the_better=True, test_threshold=20):
#     """ Find values of settings to be excluded """
#     justlen = max(len(str(v)) for v in columns) + 2
#     vallen = max(len(str(val)) for v in columns for val in master_df[v].unique()) + 2
#
#     performances = list()
#     for settings in product(*[list(master_df[var].unique()) for var in columns]):
#         criteria_values = fixed(master_df, dict(zip(columns, settings)))[criteria].mean()
#         performances.append((settings, criteria_values))
#
#     performances = sorted(performances, key=lambda _perf: _perf[1], reverse=the_larger_the_better)
#
#     if len(performances) <= test_threshold:
#         return set()
#
#     rankings = defaultdict(lambda: defaultdict(list))
#     perfvals = defaultdict(lambda: defaultdict(list))
#     _print_(''.join([str(f).center(20) for f in columns]), verbose=verbose)
#     _print_('--------------------------------------------', verbose=verbose)
#     for ranking, perf in enumerate(performances):
#         settings, f2 = perf
#         _print_(''.join([str(f).center(20) for f in settings]), end='', verbose=verbose)
#         _print_(f'{f2:.3f} -- ({ranking + 1})', verbose=verbose)
#
#         for f, s in zip(columns, settings):
#             rankings[f][s].append(ranking)
#             perfvals[f][s].append(f2)
#
#     to_remove = set()
#     lowest_pval = None
#     for col in columns:
#         keep_vals = set(master_df[col].unique())
#         for val1, val2 in combinations(master_df[col].unique(), 2):
#             # let val1 ranksum is small (better)
#             if sum(rankings[col][val1]) > sum(rankings[col][val2]):
#                 val1, val2 = val2, val1
#
#             p_val = scipy.stats.mannwhitneyu(rankings[col][val1], rankings[col][val2], alternative='less')[1]
#             # p_val = scipy.stats.ttest_ind(perfvals[col][val1], perfvals[col][val2], equal_var=False)[1]
#             if p_val < alpha:
#                 _print_(f'{str(col).rjust(justlen)}: {str(val1).rjust(vallen)} > {str(val2).ljust(vallen)} (p-value: {p_val:.3f} < {alpha})', verbose=verbose)
#                 if val2 in keep_vals:
#                     keep_vals.remove(val2)
#                 to_remove.add((col, val2))
#                 if lowest_pval is None:
#                     lowest_pval = {(col, val2, p_val)}
#                 elif next(iter(lowest_pval))[-1] == p_val:
#                     lowest_pval.add((col, val2, p_val))
#                 elif next(iter(lowest_pval))[-1] > p_val:
#                     lowest_pval = {(col, val2, p_val)}
#
#     if worst:
#         if lowest_pval is not None:
#             _print_(f"worst settings: {lowest_pval}", verbose=verbose)
#             return {(x, y) for x, y, _ in lowest_pval}
#         else:
#             return set()
#     else:
#         return to_remove
#
#
# def _header_hook(row_names=None, col_names=None, xlabel=None):
#     def inner_hook(g):
#         # post factorplot
#         if row_names:
#             for ax, row_name in zip(g.axes, row_names):
#                 ax[0].set(ylabel=row_name)
#         if col_names:
#             for ax in g.axes.flat:
#                 ax.set_title('')
#
#             for ax, col_name in zip(g.axes[0], col_names):
#                 ax.set_title(col_name)
#         if xlabel is not None:
#             for ax in g.axes[-1]:
#                 ax.set(xlabel=xlabel)
#
#     return inner_hook
#
#
# def _draw_general(master_df: pd.DataFrame, *,
#                   to_fix=None,
#                   x=None,
#                   y=None,
#                   row=None,
#                   col=None,
#                   hue=None,
#                   row_order=None,
#                   col_order=None,
#                   hue_order=None,
#                   to_melt=None,
#                   melt_id_vars=None,
#                   figsize=(6, 4),
#                   font_scale=1.4,
#                   palette=None,
#                   filename=None,
#                   y_lim=None,
#                   factor_plot_hook=None,
#                   legend_locs=(-1, -1, 4), **kwargs):
#     time.sleep(0.1)
#     print(f'  drawing: {filename}', file=sys.stderr, flush=True)
#     fixed_found = {col: master_df[col].unique()[0] for col in master_df.columns if len(master_df[col].unique()) == 1}
#     if fixed_found:
#         print('  Found Fixed::', fixed_found, file=sys.stderr, flush=True)
#         time.sleep(0.1)
#
#     # prepare data
#     working_df = master_df
#     if to_fix is not None:
#         working_df = fixed(working_df, to_fix)
#
#     if to_melt:
#         working_df = pd.melt(working_df, id_vars=melt_id_vars,
#                              value_vars=hue_order, var_name=hue, value_name=y)
#
#     # prepare
#     paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
#     sns.set(style='white', font_scale=font_scale, context='paper', rc=paper_rc)
#     plt.figure(figsize=figsize)
#     if palette is not None:
#         sns.set_palette(palette)
#
#     # prepare for options
#     factor_plot_dict = {'x': x, 'y': y, 'data': working_df}
#
#     if hue is not None:
#         factor_plot_dict.update(hue=hue)
#         if hue_order is not None:
#             factor_plot_dict.update(hue_order=hue_order)
#
#     if row is not None:
#         factor_plot_dict.update(row=row)
#         if row_order is not None:
#             factor_plot_dict.update(row_order=row_order)
#
#     if col is not None:
#         factor_plot_dict.update(col=col)
#         if col_order is not None:
#             factor_plot_dict.update(col_order=col_order)
#     if kwargs is not None:
#         factor_plot_dict.update(**kwargs)
#
#     g = sns.catplot(**factor_plot_dict)
#
#     if y_lim is not None:
#         for ax in g.axes.flat:
#             ax.set_ylim(*y_lim)
#
#     if legend_locs:
#         if legend_locs == 'out':
#             plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#         else:
#             g.axes[legend_locs[0]][legend_locs[1]].legend(loc=legend_locs[2])
#
#     if factor_plot_hook is not None:
#         if isinstance(factor_plot_hook, list):
#             for hook in factor_plot_hook:
#                 hook(g)
#
#     # saving and closing
#     sns.despine()
#     plt.tight_layout()
#     plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.02)
#     plt.close('all')
#
#
# def draw_phase_I(label: str, master_df: pd.DataFrame):
#     """ Macro average """
#     if True:
#         draw_phase_1_macro_average(label, fixed(master_df, {'order-dependence': False, 'aggregation': True}),
#                                    sns.color_palette("Paired", 6))
#
#     if True:
#         draw_order_independence(label, master_df)
#
#     if True:
#         how_aggregation(label, fixed(master_df, {'order-dependence': False, 'aggregation': True}))
#         draw_aggregation(label, master_df)
#
#     if True:
#         report_how_nonRCMC_worked(label, fixed(master_df, {'order-dependence': False, 'aggregation': True}))
#         report_how_nonRCMC_worked2(label, fixed(master_df, {'order-dependence': False, 'aggregation': True}))
#
#
# @logged
# def report_how_nonRCMC_worked2(label: str, master_df: pd.DataFrame):
#     df = master_df.copy()  # type: pd.DataFrame
#     stages = ['initial', 'recovered', 'refined']
#     for theme in stages:
#         df[f'{theme} precision'] = df[f'{theme} true positives'] / (df[f'{theme} true positives'] + df[f'{theme} false positives'])
#         df[f'{theme} recall'] = df[f'{theme} true positives'] / df['dependencies']
#         df[f'{theme} F-measure'] = 2 * df[f'{theme} precision'] * df[f'{theme} recall'] / (df[f'{theme} precision'] + df[f'{theme} recall'])
#
#     metrics = ['precision', 'recall', 'F-measure', 'true positives', 'false positives']
#
#     dfs = dict()
#     for stage_val in stages:
#         dfs[stage_val] = df.reset_index()
#         dfs[stage_val]['stage'] = stage_val
#         for rest in set(stages) - {stage_val}:
#             for metric in metrics:
#                 del dfs[stage_val][rest + ' ' + metric]
#         dfs[stage_val] = dfs[stage_val].rename(columns={stage_val + ' ' + metric: metric for metric in metrics})
#     df = pd.concat(list(dfs.values()), ignore_index=True).reindex()
#
#     print(df.columns)
#     df = df.melt(id_vars=['idx', 'base size', 'aggregation', 'order-dependence', 'stage'],
#                  value_vars=metrics,
#                  var_name='metric',
#                  value_name='y'
#                  )
#
#     _draw_general(df,
#                   x='base size', y='y', hue='stage', data=df, col='metric',
#                   hue_order=stages, col_order=metrics,
#                   palette=sns.color_palette('GnBu', 5)[2:], ci=95, n_boot=2000, sharey=False,
#                   aspect=1, size=2.5, filename=f'{working_dir}{label}/phase_I_recovery_refined.pdf', legend=False, legend_out=False,
#                   legend_locs=None, factor_plot_hook=[_header_hook(col_names=metrics), lambda g: g.axes[0][0].legend(loc=4), lambda g: [ax.set(ylabel='') for ax in g.axes.flat]]
#                   )
#
#
# @logged
# def draw_order_independence(label: str, master_df: pd.DataFrame):
#     columns_to_mean = ['initial true positives', 'initial false positives',
#                        'recovered true positives', 'recovered false positives',
#                        'refined true positives', 'refined false positives']
#     original = pd.options.display.float_format
#     pd.options.display.float_format = '{:.3f}'.format
#     print(master_df[np.logical_or(master_df['base size'] == 200, master_df['base size'] == 700)].
#           groupby(['base size', 'aggregation', 'order-dependence'])[columns_to_mean].
#           mean())
#     print(master_df.groupby(['aggregation', 'order-dependence'])[columns_to_mean].mean())
#     pd.options.display.float_format = original
#     dfs = dict()
#     stage_order = ['initial', 'recovered', 'refined']
#     for stage_val in stage_order:
#         dfs[stage_val] = master_df.reset_index()
#         dfs[stage_val]['stage'] = stage_val
#         dfs[stage_val] = dfs[stage_val].rename(columns={f'{stage_val} true positives': 'true positives', f'{stage_val} false positives': 'false positives'})
#     df = pd.concat(list(dfs.values()), ignore_index=True)
#     df['order-independence'] = np.logical_not(df['order-dependence'])
#
#     df = df.melt(id_vars=['idx', 'base size', 'aggregation', 'order-independence', 'stage'],
#                  value_vars=['true positives', 'false positives'],
#                  var_name='category',
#                  value_name='counts'
#                  )
#     row_order = ['true positives', 'false positives']
#
#     def hook(g):
#         if label == 'random':
#             for c in range(3):
#                 g.axes[0][c].set_ylim(4.8, 7.3)
#             for c in range(3):
#                 g.axes[1][c].set_ylim(0, 0.5)
#         if label == 'company':
#             for c in range(3):
#                 g.axes[0][c].set_ylim(3.4, 5)
#             for c in range(3):
#                 g.axes[1][c].set_ylim(0, 0.39)
#
#     for agg_setting in [None, True, False]:
#         if agg_setting is not None:
#             work_df = df[df['aggregation'] == agg_setting].copy()
#         else:
#             work_df = df.copy()
#         work_df['order-independence'] = work_df['order-independence'].map(lambda x: 'order-independent' if x else 'order-dependent')
#         _draw_general(work_df,
#                       x='base size', y='counts', hue='order-independence', data=work_df, row='category', col='stage',
#                       hue_order=['order-independent', 'order-dependent'], row_order=row_order, col_order=stage_order,
#                       palette=sns.color_palette('Set1', 2), ci=95, n_boot=2000, sharey=False,
#                       aspect=1.3, size=2, filename=f'{working_dir}{label}/phase_I_order_independence_agg_{agg_setting}.pdf', legend=False, legend_out=False,
#                       legend_locs=(1, 0, 1), factor_plot_hook=[_header_hook(row_order, stage_order), hook]
#                       )
#
#
# @logged
# def draw_aggregation(label: str, master_df: pd.DataFrame):
#     columns_to_mean = ['initial true positives', 'initial false positives',
#                        'recovered true positives', 'recovered false positives',
#                        'refined true positives', 'refined false positives']
#     original = pd.options.display.float_format
#     pd.options.display.float_format = '{:.3f}'.format
#     print(master_df[np.logical_or(master_df['base size'] == 200, master_df['base size'] == 700)].
#           groupby(['base size', 'order-dependence', 'aggregation'])[columns_to_mean].
#           mean())
#     print(master_df.groupby(['order-dependence', 'aggregation'])[columns_to_mean].mean())
#     pd.options.display.float_format = original
#     dfs = dict()
#     stage_order = ['initial', 'recovered', 'refined']
#     for stage_val in stage_order:
#         dfs[stage_val] = master_df.copy().reset_index()
#         dfs[stage_val]['stage'] = stage_val
#         dfs[stage_val] = dfs[stage_val].rename(columns={f'{stage_val} true positives': 'true positives', f'{stage_val} false positives': 'false positives'})
#     df = pd.concat(list(dfs.values()), ignore_index=True)
#     df['order-independence'] = np.logical_not(df['order-dependence'])
#
#     df = df.melt(id_vars=['idx', 'base size', 'aggregation', 'order-independence', 'stage'],
#                  value_vars=['true positives', 'false positives'],
#                  var_name='category',
#                  value_name='counts'
#                  )
#     row_order = ['true positives', 'false positives']
#
#     def hook(g):
#         if label == 'random':
#             for c in range(3):
#                 g.axes[0][c].set_ylim(4.8, 7.3)
#             for c in range(3):
#                 g.axes[1][c].set_ylim(0, 0.5)
#         if label == 'company':
#             for c in range(3):
#                 g.axes[0][c].set_ylim(3.4, 5)
#             for c in range(3):
#                 g.axes[1][c].set_ylim(0, 0.39)
#
#     for oi in [True, False]:
#         work_df = df[df['order-independence'] == oi].copy()
#         work_df['aggregation'] = work_df['aggregation'].map(lambda x: 'w/ aggregation' if x else 'w/o aggregation')
#         _draw_general(work_df,
#                       x='base size', y='counts', hue='aggregation', data=work_df, row='category', col='stage',
#                       hue_order=['w/ aggregation', 'w/o aggregation'], row_order=row_order, col_order=stage_order,
#                       palette=sns.color_palette('Set1', 2), ci=95, n_boot=2000, sharey=False,
#                       aspect=1.3, size=2, filename=f'{working_dir}{label}/phase_I_aggregation_with_oi_{oi}.pdf', legend=False, legend_out=False,
#                       legend_locs=(1, 0, 1), factor_plot_hook=[_header_hook(row_order, stage_order), hook]
#                       )
#
#
# @logged
# def report_how_nonRCMC_worked(label: str, master_df: pd.DataFrame):
#     df = master_df.copy()  # type: pd.DataFrame
#     del df['idx']
#     del df['aggregation']
#     del df['order-dependence']
#     del df['right direction (aggregated)']
#     del df['reverse direction (aggregated)']
#     del df['false positives (aggregated)']
#     original = pd.options.display.float_format
#     pd.options.display.float_format = '{:.2f}'.format
#     print(f'========= mean ({label})=========')
#     print(df.mean())
#     print("df.groupby('base size').mean()")
#     print(df.groupby('base size').mean())
#
#     pd.options.display.float_format = original
#
#     df = fixed(master_df, {'aggregation': True})
#     df = df.mean()
#
#     delta_TP = df['recovered true positives'] - df['initial true positives']
#     delta_FP = df['recovered false positives'] - df['initial false positives']
#     precision = delta_TP / (delta_TP + delta_FP)
#     missing = (df['dependencies'] - df['initial true positives'])
#
#     print(f"On average there are {missing:.2f} missing relational dependencies (with total {df['dependencies']:.2f}).")
#     print(f"non-RCMC recovered {delta_TP + delta_FP:.2f} dependencies ({100 * (delta_TP + delta_FP) / missing:.1f}%).")
#     print(f"Among them, {delta_TP:.2f} is correctly recovered while {delta_FP:.2f} is wrongly recovered.")
#     print(f"which is, {precision:.4f} (precision).")
#
#     delta_TP = df['refined true positives'] - df['initial true positives']
#     delta_FP = df['refined false positives'] - df['initial false positives']
#
#     first_restored = (df['recovered true positives'] - df['initial true positives']) + (df['recovered false positives'] - df['initial false positives'])
#     finally_restored = delta_TP + delta_FP
#     putting_backed = first_restored - finally_restored
#     print(f"\nWith further refining, {delta_TP + delta_FP:.2f} dependencies are finally recovered.")
#     print(f"That is {100 * putting_backed / first_restored:.1f}% of recovered dependencies is removed again.")
#     print(f"Among them, {delta_TP:.2f} dependencies are correct while {delta_FP:.2f} dependencies are wrong.")
#
#     # refining quality
#     delta_FN = df['recovered true positives'] - df['refined true positives']
#     delta_TN = df['recovered false positives'] - df['refined false positives']
#     precision = delta_TN / (delta_FN + delta_TN)
#     print(f"{delta_FN:.4f}: why do you put back, it was FN --> TP --> FN")
#     print(f"{delta_TN:.4f}: you put back right, it was TN --> FP --> TN")
#     print(f'{precision:.4f}: okay refinement ... is about this precision')
#
#
# @logged
# def how_aggregation(label: str, master_df: pd.DataFrame):
#     df = master_df.rename(columns={'right direction (aggregated)': 'right direction',
#                                    'reverse direction (aggregated)': 'reverse direction',
#                                    'false positives (aggregated)': 'false positive'})
#     df['counts'] = df['right direction'] + df['reverse direction'] + df['false positive']
#     how_aggregated_worked = list()
#     for size in sorted(df['base size'].unique()):
#         subdf = fixed(df, {'base size': size})
#         counts = subdf['right direction'].sum() + subdf['reverse direction'].sum() + subdf['false positive'].sum()
#         how_aggregated_worked.append([size, counts, subdf['right direction'].mean(), subdf['reverse direction'].mean(), subdf['false positive'].mean()])
#
#     agg_df = pd.DataFrame(how_aggregated_worked, columns=['base size', 'counts', 'right direction', 'reverse direction', 'false positive'])
#     draw_how_aggregated_works(label, agg_df, df)
#
#
# @logged
# def draw_how_aggregated_works(label: str, master_df: pd.DataFrame, df2):
#     paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
#     sns.set(style='white', font_scale=1.4, context='paper', rc=paper_rc)
#     sns.set_palette(sns.color_palette("Set2"))
#     fig, axes = plt.subplots(1, 2, figsize=(7, 3))
#
#     left, right = 0, 1
#     subdf = master_df.copy()  # type: pd.DataFrame
#
#     print("=== Average counts ======================")
#     print(df2.groupby('base size')[['right direction', 'reverse direction', 'false positive']].mean())
#     """ Left plot """
#     df = pd.melt(df2, 'base size', ['right direction', 'reverse direction', 'false positive'], var_name='types', value_name='average count')
#     ax = axes[left]
#     sns.pointplot('base size', 'average count', hue='types', data=df, ax=ax, dodge=True)
#     ax.set_ylim(bottom=0.0)
#     ax.set_title('counts')
#     ax.legend(title='')
#     ax.set_xlabel('base size')
#     ax.set_ylabel('')
#
#     """ Right plot """
#     ax = axes[right]
#     cols = ['right direction', 'reverse direction', 'false positive']
#     normalizer = subdf[['right direction', 'reverse direction', 'false positive']].sum(axis=1)
#     for col in cols:
#         subdf[col] = subdf[col] / normalizer
#     stds_df = subdf.copy()  # type: pd.DataFrame
#     for col in cols:
#         stds_df[col] = np.sqrt((subdf[col] * (1 - subdf[col])) / subdf['counts'])
#     before_melt_subdf = subdf.copy()
#     subdf = pd.melt(subdf, 'base size', ['right direction', 'reverse direction', 'false positive'], var_name='types', value_name='ratios')
#     stds_df = pd.melt(stds_df, 'base size', value_vars=['right direction', 'reverse direction', 'false positive'], var_name='types', value_name='error')
#     subdf['error'] = 1.96 * stds_df['error']
#     df = subdf
#     #
#     u = sorted(df['base size'].unique())
#     x = np.arange(len(u))
#     subx = df['types'].unique()
#     bottom = None
#     for i, gr in enumerate(subx):
#         dfg = df[df['types'] == gr]
#         if i == 0:
#             ax.bar(x, dfg['ratios'].values, label=f"{gr}", yerr=dfg['error'].values)
#             bottom = dfg['ratios'].values
#         else:
#             ax.bar(x, dfg['ratios'].values, bottom=bottom, label=f"{gr}", yerr=dfg['error'].values)
#             bottom += dfg['ratios'].values
#     ax.set_xlabel('base size')
#     ax.yaxis.tick_right()
#     ax.set_title('proportions')
#     ax.set_xticks(x)
#     ax.set_xticklabels(u)
#     ax.legend().set_visible(False)
#     ax.set_ylim(0.0, 1.1)
#     print("=== Proportions ======================")
#     print(before_melt_subdf)
#     # for col in ['right direction', 'reverse direction', 'false positive']:
#     #     to_print[col] = (100 * to_print[col]).map('{:.1f}%'.format)
#     # print(to_print[['base size', 'right direction', 'reverse direction', 'false positive']])
#     sns.despine()
#     plt.tight_layout()
#     plt.savefig(f'{working_dir}{label}/phase_I_aggregation.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
#     plt.close()
#
#
# @logged
# def draw_phase_1_macro_average(label: str, master_df: pd.DataFrame, pal):
#     df = fixed(master_df, {'aggregation': True})
#
#     for theme in ['initial', 'refined']:
#         df[f'{theme} precision'] = df[f'{theme} true positives'] / (df[f'{theme} true positives'] + df[f'{theme} false positives'])
#         df[f'{theme} recall'] = df[f'{theme} true positives'] / df['dependencies']
#         df[f'{theme} F-measure'] = 2 * df[f'{theme} precision'] * df[f'{theme} recall'] / (df[f'{theme} precision'] + df[f'{theme} recall'])
#
#     hue_order = [theme + ' ' + metric for metric in ['precision', 'recall', 'F-measure'] for theme in ['initial', 'refined']]
#
#     for legend in [False, True]:
#         _draw_general(df, x='base size', y='ratio', hue='metric', to_melt=True, melt_id_vars=['idx', 'base size', 'aggregation'],
#                       hue_order=hue_order,
#                       filename=f'{working_dir}{label}/macro_average_legend_{legend}.pdf', palette=pal,
#                       legend_locs='out' if legend else None,
#                       legend=False, legend_out=False, size=3, aspect=1.3, dodge=True
#                       )
#
#     return df
#
#
# @logged
# def check_number_of_tests(label: str, master_df: pd.DataFrame):
#     working_df = fixed(master_df, {'aggregation': False, 'orientation rule': 'majority'})
#     working_df = working_df[np.logical_or(working_df['base size'] == 200, working_df['base size'] == 700)]
#
#     working_df['detect'] = np.logical_and(working_df['detect rbo'], working_df['detect post rbo'])
#     working_df = working_df[working_df['detect rbo'] == working_df['detect post rbo']]  # both on or both of
#     working_df['number of tests'] = (working_df['RBO test'] + working_df['post-RBO test'])
#
#     working_df["criteria"] = working_df["sepset rule"] + ' ' + working_df["detect"].map(lambda x: 'w/ detect' if x else 'w/o detect')
#
#     hue_order = [sr + ' ' + det for sr in ['first', 'minimal', 'full'] for det in ['w/o detect', 'w/ detect']]
#     paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
#     sns.set(style='white', font_scale=1.6, palette=sns.color_palette('Paired', 6), context='paper', rc=paper_rc)
#     plt.figure()
#     g = sns.catplot(x='base size',
#                     y='number of tests',
#                     hue='criteria',
#                     data=working_df,
#                     hue_order=hue_order,
#                     kind='box', legend=False, legend_out=False,
#                     size=3.25, aspect=1.2
#                     )
#     if label == 'random':
#         plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.)
#     g.axes.flat[0].set_yscale('log', basey=2)
#     sns.despine()
#     plt.tight_layout()
#     plt.savefig(f'{working_dir}{label}/number_of_tests.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
#     plt.close('all')
#
#
# @logged
# def check_number_of_cases(label: str, master_df: pd.DataFrame):
#     working_df = fixed(master_df, {'aggregation': False, 'orientation rule': 'majority'})
#     working_df = working_df[np.logical_or(working_df['base size'] == 200, working_df['base size'] == 700)]
#
#     working_df['detect'] = np.logical_and(working_df['detect rbo'], working_df['detect post rbo'])
#     working_df = working_df[working_df['detect rbo'] == working_df['detect post rbo']]  # both on or both of
#     working_df['number of cases'] = (working_df['RBO case'] + working_df['post-RBO case'])
#
#     working_df["criteria"] = working_df["sepset rule"] + ' ' + working_df["detect"].map(lambda x: 'w/ detect' if x else 'w/o detect')
#
#     hue_order = [sr + ' ' + det for sr in ['first', 'minimal', 'full'] for det in ['w/o detect', 'w/ detect']]
#     paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
#     sns.set(style='white', font_scale=1.4, palette=sns.color_palette('Paired', 6), context='paper', rc=paper_rc)
#     plt.figure()
#     sns.catplot(x='base size',
#                 y='number of cases',
#                 hue='criteria',
#                 data=working_df,
#                 hue_order=hue_order,
#                 dodge=True,
#                 kind='strip',
#                 jitter=1,
#                 alpha=0.5,
#                 )
#     sns.despine()
#     plt.tight_layout()
#     plt.savefig(f'{working_dir}{label}/number_of_cases.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
#     plt.close('all')
#
#
# def draw_phase_II(label: str, master_df: pd.DataFrame):
#     master_df['precision'] = float('nan')
#     safe = master_df['directed'] != 0
#     master_df.loc[safe, 'precision'] = master_df.loc[safe, 'correct directed'] / master_df.loc[safe, 'directed']
#     master_df['recall'] = master_df['correct directed'] / master_df['directables']
#     master_df['F-measure'] = 2 * master_df['precision'] * master_df['recall'] / (master_df['precision'] + master_df['recall'])
#
#     if True:
#         perform_check_best_settings(label, master_df, verbose=True, seed=10)
#
#     if True:
#         check_splits(label, master_df)
#         check_splits2(label, master_df)
#
#     if True:
#         check_number_of_tests(label, master_df)
#
#
# @logged
# @reproducible
# def perform_check_best_settings(label: str, master_df: pd.DataFrame, verbose=False):
#     # not to choose base size
#     fixables = sorted(['base size', 'aggregation', 'sepset rule', 'orientation rule', 'detect rbo', 'detect post rbo'])
#     known_order = {'orientation rule': ['majority', 'conservative'], 'sepset rule': ['first', 'minimal', 'full']}
#
#     for col in fixables:
#         _print_(col, verbose=verbose)
#         for val in known_order[col] if col in known_order else sorted(master_df[col].unique()):
#             working_df = fixed(master_df, {col: val})
#
#             boot_precision = np.zeros((2000,))
#             boot_recall = np.zeros((2000,))
#             boot_f = np.zeros((2000,))
#             for boot in range(2000):
#                 sampling = np.random.randint(len(working_df), size=len(working_df))
#                 subsampled = working_df.iloc[sampling]
#                 precision = subsampled['correct directed'].sum() / subsampled['directed'].sum()
#                 recall = subsampled['correct directed'].sum() / subsampled['directables'].sum()
#                 f_measure = 2 * precision * recall / (precision + recall)
#                 boot_precision[boot] = precision
#                 boot_recall[boot] = recall
#                 boot_f[boot] = f_measure
#
#             precision = working_df['correct directed'].sum() / working_df['directed'].sum()
#             recall = working_df['correct directed'].sum() / working_df['directables'].sum()
#             f_measure = 2 * precision * recall / (precision + recall)
#
#             low_p, high_p = np.percentile(boot_precision, [5, 95])
#             low_r, high_r = np.percentile(boot_recall, [5, 95])
#             low_f, high_f = np.percentile(boot_f, [5, 95])
#             _print_(f'    {str(val).rjust(15)}  {precision:.4f} ({low_p:.4f} ~ {high_p:.4f})  {recall:.4f} ({low_r:.4f} ~ {high_r:.4f})  {f_measure:.4f} ({low_f:.4f} ~ {high_f:.4f})',
#                     verbose=verbose)
#
#     macro = list()
#     for setting in product(*[list(master_df[var].unique()) for var in fixables]):
#         subsampled = fixed(master_df, dict(zip(fixables, setting)))
#         precision = subsampled['correct directed'].sum() / subsampled['directed'].sum()
#         recall = subsampled['correct directed'].sum() / subsampled['directables'].sum()
#         f_measure = 2 * precision * recall / (precision + recall)
#         macro.append([*setting, precision, recall, f_measure])
#
#     newdf = pd.DataFrame(macro, columns=[*fixables, 'precision', 'recall', 'F-measure'])
#     _print_('precision', verbose=verbose)
#     _print_(newdf.sort_values('precision', ascending=False).iloc[:10], verbose=verbose)
#     _print_('recall', verbose=verbose)
#     _print_(newdf.sort_values('recall', ascending=False).iloc[:10], verbose=verbose)
#     _print_('F-measure', verbose=verbose)
#     _print_(newdf.sort_values('F-measure', ascending=False).iloc[:10], verbose=verbose)
#
#     performances = np.zeros((len(newdf), 4))
#     performances[:, 0] = newdf['precision']
#     performances[:, 1] = newdf['recall']
#     performances[:, 2] = newdf['F-measure']
#     performances[:, 3] = newdf['base size']
#     # dumb
#     to_retain = list()
#     for i in range(len(performances)):
#         for j in range(len(performances)):
#             if i == j:
#                 continue
#             if performances[i, 0] < performances[j, 0] and performances[i, 1] < performances[j, 1] and performances[i, 3] == performances[j, 3]:
#                 break
#         else:
#             to_retain.append(i)
#     best_performances = performances[to_retain, :]
#
#     colors = {100 * (i + 2): c for i, c in enumerate(sns.color_palette("YlGn", 9)[3:])}
#     scatter_colors = [colors[size] for size in performances[:, 3]]
#
#     paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
#     sns.set(style='white', font_scale=1.4, context='paper', rc=paper_rc)
#     plt.figure(figsize=(4, 4))
#     plt.scatter(performances[:, 0], performances[:, 1],
#                 c=scatter_colors,
#                 alpha=0.6,
#                 lw=0,
#                 s=25)
#
#     best_df = pd.DataFrame(best_performances, columns=['precision', 'recall', 'F-measure', 'base size'])
#     for key, gdf in best_df.groupby('base size'):
#         sortedf = gdf.sort_values(['precision', 'recall'])
#         plt.gca().plot(sortedf['precision'], sortedf['recall'], color=colors[key], label=int(key))
#
#     if label == 'random':
#         plt.gca().legend()
#     if label == 'random':
#         plt.gca().set_xlim(0.75, 1.0)
#         plt.gca().set_ylim(0.1, 0.85)
#     else:
#         plt.gca().set_xlim(0.8, 1.0)
#         plt.gca().set_ylim(0.3, 0.95)
#     plt.gca().set_xlabel('precision')
#     plt.gca().set_ylabel('recall')
#     sns.despine()
#     plt.tight_layout()
#     plt.savefig(f'{working_dir}{label}/settings_phase_II.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
#     plt.close('all')
#
#
# @logged
# def check_splits(label: str, master_df: pd.DataFrame):
#     for sepset_rule in ['first', 'minimal', 'full']:
#         paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
#         sns.set(style='white', font_scale=1.4, context='paper', rc=paper_rc)
#
#         col_fail_noncollider, col_noncollider, col_fail_collider, col_collider = sns.color_palette('Paired', 4)
#         col_weak = sns.color_palette("Greys", 6)[2]
#         col_weak2 = sns.color_palette("Greys", 6)[3]
#         col_rest = sns.color_palette("Greys", 6)[-1]
#
#         fig, axes = plt.subplots(2, 2, figsize=(6, 5))
#         axes[0][0].set_title('RBO')
#         axes[0][1].set_title('non-RBO')
#         for col, sublabel in enumerate(['RBO', 'post-RBO']):
#             working_df = fixed(master_df, {f'detect rbo': True, 'detect post rbo': True, 'sepset rule': sepset_rule})  # first?
#             working_df['weak'] = working_df[f'{sublabel} violation']
#             working_df['correct non-collider'] = working_df[f'{sublabel} correct violation-collider']
#             working_df['wrong non-collider'] = working_df[f'{sublabel} violation-collider'] - working_df[f'{sublabel} correct violation-collider']
#             working_df['correct collider'] = working_df[f'{sublabel} correct violation-non-collider']
#             working_df['wrong collider'] = working_df[f'{sublabel} violation-non-collider'] - working_df[f'{sublabel} correct violation-non-collider']
#             working_df['normal test case'] = working_df[f'{sublabel} case'] - working_df['weak'] - working_df[f'{sublabel} violation-non-collider'] - working_df[f'{sublabel} violation-collider']
#             working_df = working_df.groupby(['base size']).mean().reset_index()
#             working_df = pd.melt(working_df, id_vars=['base size'],
#                                  value_vars=['weak', 'correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider', 'normal test case'],
#                                  var_name='detection type',
#                                  value_name='counts')
#
#             df = working_df
#             colors = [col_weak, col_noncollider, col_fail_noncollider, col_collider, col_fail_collider, col_rest]
#             ax = axes[0][col]
#
#             margin_bottom = np.zeros(len(working_df['base size'].unique()))
#             for num, detection_type in enumerate(['weak', 'correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider', 'normal test case']):
#                 values = list(df[df['detection type'] == detection_type].loc[:, 'counts'])
#
#                 df[df['detection type'] == detection_type].plot.bar(x='base size', y='counts', ax=ax, stacked=True,
#                                                                     bottom=margin_bottom, color=colors[num], label=detection_type)
#                 margin_bottom += values
#
#             if col == 1:
#                 handles, labels = ax.get_legend_handles_labels()
#                 ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
#             else:
#                 ax.legend().remove()
#
#             working_df = fixed(master_df, {f'detect rbo': True, 'detect post rbo': True, 'sepset rule': sepset_rule})  # first?
#             working_df['correct non-collider'] = working_df[f'{sublabel} correct non-collider']
#             working_df['wrong non-collider'] = working_df[f'{sublabel} non-collider'] - working_df[f'{sublabel} correct non-collider']
#             working_df['correct collider'] = working_df[f'{sublabel} correct collider']
#             working_df['wrong collider'] = working_df[f'{sublabel} collider'] - working_df[f'{sublabel} correct collider']
#             working_df['correct collider-fail'] = working_df[f'{sublabel} correct collider-fail']
#             working_df['wrong collider-fail'] = working_df[f'{sublabel} collider-fail'] - working_df[f'{sublabel} correct collider-fail']
#             working_df = working_df.groupby(['base size']).mean().reset_index()
#             working_df = pd.melt(working_df, id_vars=['base size'],
#                                  value_vars=['correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider', 'correct collider-fail', 'wrong collider-fail'],
#                                  var_name='detection type',
#                                  value_name='counts')
#
#             df = working_df
#             colors = [col_noncollider, col_fail_noncollider, col_collider, col_fail_collider, col_weak2, col_weak]
#             ax = axes[1][col]
#             margin_bottom = np.zeros(len(working_df['base size'].unique()))
#             for num, detection_type in enumerate(['correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider', 'correct collider-fail', 'wrong collider-fail']):
#                 values = list(df[df['detection type'] == detection_type].loc[:, 'counts'])
#
#                 df[df['detection type'] == detection_type].plot.bar(x='base size', y='counts', ax=ax, stacked=True,
#                                                                     bottom=margin_bottom, color=colors[num], label=detection_type)
#                 margin_bottom += values
#
#             if col == 1:
#                 handles, labels = ax.get_legend_handles_labels()
#                 ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
#             else:
#                 ax.legend().remove()
#
#         _, max1 = axes[0][0].get_ylim()
#         _, max2 = axes[0][1].get_ylim()
#         axes[0][0].set_ylim(0, max(max1, max2))
#         axes[0][1].set_ylim(0, max(max1, max2))
#
#         _, max1 = axes[1][0].get_ylim()
#         _, max2 = axes[1][1].get_ylim()
#         axes[1][0].set_ylim(0, max(max1, max2))
#         axes[1][1].set_ylim(0, max(max1, max2))
#
#         axes[0][0].set_xticks([])
#         axes[0][0].set_xticklabels([])
#         axes[0][0].set_xlabel('')
#         axes[0][1].set_xticks([])
#         axes[0][1].set_xticklabels([])
#         axes[0][1].set_xlabel('')
#
#         axes[0][1].set_yticks([])
#         axes[0][1].set_yticklabels([])
#         axes[1][1].set_yticks([])
#         axes[1][1].set_yticklabels([])
#
#         axes[0][0].set_ylabel('detection cases')
#         axes[1][0].set_ylabel('normal tests')
#
#         sns.despine()
#         plt.tight_layout()
#         plt.savefig(f'{working_dir}{label}/2-by-2-rbo-post-rbo-statistics_{sepset_rule}.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
#         plt.close('all')
#
#
# @logged
# def check_splits2(label: str, master_df: pd.DataFrame):
#     for sepset_rule in ['first', 'minimal', 'full']:
#         paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
#         sns.set(style='white', font_scale=1.4, context='paper', rc=paper_rc)
#
#         col_fail_noncollider, col_noncollider, col_fail_collider, col_collider = sns.color_palette('Paired', 4)
#         col_weak = sns.color_palette("Greys", 6)[2]
#         col_weak2 = sns.color_palette("Greys", 6)[3]
#
#         fig, axes = plt.subplots(1, 2, figsize=(6, 3))
#         axes[0].set_title('RBO')
#         axes[1].set_title('non-RBO')
#         for col, sublabel in enumerate(['RBO', 'post-RBO']):
#
#             working_df = fixed(master_df, {f'detect rbo': False, 'detect post rbo': False, 'sepset rule': sepset_rule})  # first?
#             working_df['correct non-collider'] = working_df[f'{sublabel} correct non-collider']
#             working_df['wrong non-collider'] = working_df[f'{sublabel} non-collider'] - working_df[f'{sublabel} correct non-collider']
#             working_df['correct collider'] = working_df[f'{sublabel} correct collider']
#             working_df['wrong collider'] = working_df[f'{sublabel} collider'] - working_df[f'{sublabel} correct collider']
#             working_df['correct collider-fail'] = working_df[f'{sublabel} correct collider-fail']
#             working_df['wrong collider-fail'] = working_df[f'{sublabel} collider-fail'] - working_df[f'{sublabel} correct collider-fail']
#             working_df = working_df.groupby(['base size']).mean().reset_index()
#             working_df = pd.melt(working_df, id_vars=['base size'],
#                                  value_vars=['correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider', 'correct collider-fail', 'wrong collider-fail'],
#                                  var_name='detection type',
#                                  value_name='counts')
#
#             df = working_df
#             colors = [col_noncollider, col_fail_noncollider, col_collider, col_fail_collider, col_weak2, col_weak]
#             ax = axes[col]
#             margin_bottom = np.zeros(len(working_df['base size'].unique()))
#             for num, detection_type in enumerate(['correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider']):
#                 values = list(df[df['detection type'] == detection_type].loc[:, 'counts'])
#
#                 df[df['detection type'] == detection_type].plot.bar(x='base size', y='counts', ax=ax, stacked=True,
#                                                                     bottom=margin_bottom, color=colors[num], label=detection_type)
#                 margin_bottom += values
#
#             if col == 1:
#                 handles, labels = ax.get_legend_handles_labels()
#                 ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
#             else:
#                 ax.legend().remove()
#
#         _, max1 = axes[0].get_ylim()
#         _, max2 = axes[1].get_ylim()
#         axes[0].set_ylim(0, max(max1, max2))
#         axes[1].set_ylim(0, max(max1, max2))
#
#         axes[1].set_yticks([])
#         axes[1].set_yticklabels([])
#
#         axes[0].set_ylabel('normal tests')
#
#         sns.despine()
#         plt.tight_layout()
#         plt.savefig(f'{working_dir}{label}/1-by-2-rbo-post-rbo-statistics-no-detect_{sepset_rule}.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
#         plt.close('all')
#
#
# def read_master_dfs(label: str):
#     """ Read csv files into DataFrame for given data type both phase I and II. """
#     master_dfs = dict()
#
#     master_dfs[1] = pd.read_csv(f'{working_dir}{label}/phase_1.csv.zip', header=None, compression='zip')
#     master_dfs[1].columns = list(phase_1_columns)
#
#     master_dfs[2] = pd.read_csv(f'{working_dir}{label}/phase_2.csv.zip', header=None, compression='zip')
#     master_dfs[2].columns = list(phase_2_columns)
#     del master_dfs[2]['dummy0']
#     del master_dfs[2]['dummy1']
#     del master_dfs[2]['dummy2']
#
#     return master_dfs
#
#
# def main(argv):
#     for label in {'company', 'random'}:
#         if label not in argv:
#             continue
#         master_dfs = read_master_dfs(label)
#
#         draw_phase_I(label, master_dfs[1])
#         draw_phase_II(label, master_dfs[2])
#
#
# if __name__ == '__main__':
#     pd.set_option('expand_frame_repr', False)
#     main(list(sys.argv) + ['random', 'company'])
