import warnings
from collections import defaultdict
from itertools import product, combinations

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import rc
from scipy.stats import ttest_ind

from rrpcd._UAI_2019.unified_analyze_utils import logged, _print_
from rrpcd.experiments.exp_utils import fixed
from rrpcd.experiments.unified_evaluation import stats_keys
from rrpcd.utils import reproducible

warnings.filterwarnings("error")
# Pycharm debugger uses where script exists ...
# working_dir = f''
working_dir = f'rrpcd/_UAI_2019/'

phase_2_id_vars = ('idx', 'base size', 'aggregation', 'sepset rule', 'orientation rule', 'detect rbo', 'detect post rbo')
phase_2_columns = (*phase_2_id_vars, 'dummy0',
                   *['RBO ' + k for k in stats_keys()], 'dummy1',
                   *['post-RBO ' + k for k in stats_keys()], 'dummy2',
                   'correct directed', 'directed', 'directables'
                   )


def _find_worse_settings(master_df, criteria, columns, alpha=0.01, verbose=False, worst=False, the_larger_the_better=True, test_threshold=20):
    """ Find values of settings to be excluded """
    justlen = max(len(str(v)) for v in columns) + 2
    vallen = max(len(str(val)) for v in columns for val in master_df[v].unique()) + 2

    performances = list()
    for settings in product(*[list(master_df[var].unique()) for var in columns]):
        criteria_values = fixed(master_df, dict(zip(columns, settings)))[criteria].mean()
        performances.append((settings, criteria_values))

    performances = sorted(performances, key=lambda _perf: _perf[1], reverse=the_larger_the_better)

    if len(performances) <= test_threshold:
        return set()

    rankings = defaultdict(lambda: defaultdict(list))
    perfvals = defaultdict(lambda: defaultdict(list))
    _print_(''.join([str(f).center(20) for f in columns]), verbose=verbose)
    _print_('--------------------------------------------', verbose=verbose)
    for ranking, perf in enumerate(performances):
        settings, f2 = perf
        _print_(''.join([str(f).center(20) for f in settings]), end='', verbose=verbose)
        _print_(f'{f2:.3f} -- ({ranking + 1})', verbose=verbose)

        for f, s in zip(columns, settings):
            rankings[f][s].append(ranking)
            perfvals[f][s].append(f2)

    to_remove = set()
    lowest_pval = None
    for col in columns:
        keep_vals = set(master_df[col].unique())
        for val1, val2 in combinations(master_df[col].unique(), 2):
            # let val1 ranksum is small (better)
            if sum(rankings[col][val1]) > sum(rankings[col][val2]):
                val1, val2 = val2, val1

            p_val = scipy.stats.mannwhitneyu(rankings[col][val1], rankings[col][val2], alternative='less')[1]
            # p_val = scipy.stats.ttest_ind(perfvals[col][val1], perfvals[col][val2], equal_var=False)[1]
            if p_val < alpha:
                _print_(f'{str(col).rjust(justlen)}: {str(val1).rjust(vallen)} > {str(val2).ljust(vallen)} (p-value: {p_val:.3f} < {alpha})', verbose=verbose)
                if val2 in keep_vals:
                    keep_vals.remove(val2)
                to_remove.add((col, val2))
                if lowest_pval is None:
                    lowest_pval = {(col, val2, p_val)}
                elif next(iter(lowest_pval))[-1] == p_val:
                    lowest_pval.add((col, val2, p_val))
                elif next(iter(lowest_pval))[-1] > p_val:
                    lowest_pval = {(col, val2, p_val)}

    if worst:
        if lowest_pval is not None:
            _print_(f"worst settings: {lowest_pval}", verbose=verbose)
            return {(x, y) for x, y, _ in lowest_pval}
        else:
            return set()
    else:
        return to_remove


# @logged
def check_number_of_tests(folder, master_df: pd.DataFrame):
    working_df = fixed(master_df, {'aggregation': False, 'orientation rule': 'majority'})
    working_df = working_df[np.logical_or(working_df['base size'] == 200, working_df['base size'] == 500)]

    working_df['detect'] = np.logical_and(working_df['detect rbo'], working_df['detect post rbo'])
    working_df = working_df[working_df['detect rbo'] == working_df['detect post rbo']]  # both on or both of
    working_df['number of tests'] = (working_df['RBO test'] + working_df['post-RBO test'])

    working_df["criteria"] = working_df["sepset rule"] + ' ' + working_df["detect"].map(lambda x: 'w/ detect' if x else 'w/o detect')

    hue_order = [sr + ' ' + det for sr in ['first', 'minimal', 'full'] for det in ['w/o detect', 'w/ detect']]
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
    sns.set(style='white', font_scale=1.2, palette=sns.color_palette('Paired', 6), context='paper', rc=paper_rc, color_codes=False)
    plt.figure()
    g = sns.catplot(x='base size',
                    y='number of tests',
                    hue='criteria',
                    data=working_df,
                    hue_order=hue_order,
                    kind='box', legend=False, legend_out=False,
                    height=3.25, aspect=1.2
                    )

    plt.legend(bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.)
    g.axes.flat[0].set_yscale('log', basey=2)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{working_dir}{folder}/number_of_tests.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close('all')


# @logged
def check_number_of_cases(folder, master_df: pd.DataFrame):
    working_df = fixed(master_df, {'aggregation': False, 'orientation rule': 'majority'})
    working_df = working_df[np.logical_or(working_df['base size'] == 200, working_df['base size'] == 500)]

    working_df['detect'] = np.logical_and(working_df['detect rbo'], working_df['detect post rbo'])
    working_df = working_df[working_df['detect rbo'] == working_df['detect post rbo']]  # both on or both of
    working_df['number of cases'] = (working_df['RBO case'] + working_df['post-RBO case'])

    working_df["criteria"] = working_df["sepset rule"] + ' ' + working_df["detect"].map(lambda x: 'w/ detect' if x else 'w/o detect')

    hue_order = [sr + ' ' + det for sr in ['first', 'minimal', 'full'] for det in ['w/o detect', 'w/ detect']]
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
    sns.set(style='white', font_scale=1.2, palette=sns.color_palette('Paired', 6), context='paper', rc=paper_rc)
    plt.figure()
    sns.catplot(x='base size',
                y='number of cases',
                hue='criteria',
                data=working_df,
                hue_order=hue_order,
                dodge=True,
                kind='strip',
                jitter=1,
                alpha=0.5,
                )
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{working_dir}{folder}/number_of_cases.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close('all')


def draw_phase_II(label: str, folder, master_df: pd.DataFrame):
    master_df['precision'] = float('nan')
    safe = master_df['directed'] != 0
    master_df.loc[safe, 'precision'] = master_df.loc[safe, 'correct directed'] / master_df.loc[safe, 'directed']
    master_df['recall'] = master_df['correct directed'] / master_df['directables']
    master_df['F-measure'] = 2 * master_df['precision'] * master_df['recall'] / (master_df['precision'] + master_df['recall'])

    if True:
        perform_check_best_settings(label, folder, master_df, verbose=True, seed=10)

    if False:
        check_splits(folder, master_df)
        check_splits2(folder, master_df)

    if False:
        check_number_of_tests(folder, master_df)


# @logged
@reproducible
def perform_check_best_settings(label: str, folder, master_df: pd.DataFrame, verbose=False):
    # not to choose base size
    fixables = sorted(['base size', 'aggregation', 'sepset rule', 'orientation rule', 'detect rbo', 'detect post rbo'])
    # known_order = {'orientation rule': ['majority', 'conservative'], 'sepset rule': ['first', 'minimal', 'full']}
    known_order = {'orientation rule': ['majority'], 'sepset rule': ['minimal']}

    for col in fixables:
        _print_(col, verbose=verbose)
        for val in known_order[col] if col in known_order else sorted(master_df[col].unique()):
            working_df = fixed(master_df, {col: val})

            boot_precision = np.zeros((2000,))
            boot_recall = np.zeros((2000,))
            boot_f = np.zeros((2000,))
            for boot in range(2000):
                sampling = np.random.randint(len(working_df), size=len(working_df))
                subsampled = working_df.iloc[sampling]
                precision = subsampled['correct directed'].sum() / (subsampled['directed'].sum() + 1e-20)
                recall = subsampled['correct directed'].sum() / subsampled['directables'].sum()
                f_measure = 2 * precision * recall / (precision + recall)
                boot_precision[boot] = precision
                boot_recall[boot] = recall
                boot_f[boot] = f_measure

            precision = working_df['correct directed'].sum() / working_df['directed'].sum()
            recall = working_df['correct directed'].sum() / working_df['directables'].sum()
            f_measure = 2 * precision * recall / (precision + recall)

            low_p, high_p = np.percentile(boot_precision, [5, 95])
            low_r, high_r = np.percentile(boot_recall, [5, 95])
            low_f, high_f = np.percentile(boot_f, [5, 95])
            _print_(f'    {str(val).rjust(15)}  {precision:.4f} ({low_p:.4f} ~ {high_p:.4f})  {recall:.4f} ({low_r:.4f} ~ {high_r:.4f})  {f_measure:.4f} ({low_f:.4f} ~ {high_f:.4f})',
                    verbose=verbose)

    macro = list()
    for setting in product(*[list(master_df[var].unique()) for var in fixables]):
        subsampled = fixed(master_df, dict(zip(fixables, setting)))
        if len(subsampled) == 0:
            continue
        precision = subsampled['correct directed'].sum() / subsampled['directed'].sum()
        recall = subsampled['correct directed'].sum() / subsampled['directables'].sum()
        f_measure = 2 * precision * recall / (precision + recall)
        macro.append([*setting, precision, recall, f_measure])

    newdf = pd.DataFrame(macro, columns=[*fixables, 'precision', 'recall', 'F-measure'])
    print(newdf)
    _print_('precision', verbose=verbose)
    _print_(newdf.sort_values('precision', ascending=False).iloc[:10], verbose=verbose)
    _print_('recall', verbose=verbose)
    _print_(newdf.sort_values('recall', ascending=False).iloc[:10], verbose=verbose)
    _print_('F-measure', verbose=verbose)
    _print_(newdf.sort_values('F-measure', ascending=False).iloc[:10], verbose=verbose)

    performances = np.zeros((len(newdf), 4))
    performances[:, 0] = newdf['precision']
    performances[:, 1] = newdf['recall']
    performances[:, 2] = newdf['F-measure']
    performances[:, 3] = newdf['base size']
    # dumb
    to_retain = list()
    for i in range(len(performances)):
        for j in range(len(performances)):
            if i == j:
                continue
            if performances[i, 0] < performances[j, 0] and performances[i, 1] < performances[j, 1] and performances[i, 3] == performances[j, 3]:
                break
        else:
            to_retain.append(i)
    best_performances = performances[to_retain, :]

    colors = {100 * (i + 2): c for i, c in enumerate(sns.color_palette("YlGn", 9)[3:])}
    scatter_colors = [colors[size] for size in performances[:, 3]]

    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
    sns.set(style='white', font_scale=1.2, context='paper', rc=paper_rc)
    plt.figure(figsize=(4, 4))
    plt.scatter(performances[:, 0], performances[:, 1],
                c=scatter_colors,
                alpha=0.6,
                lw=0,
                s=25)

    best_df = pd.DataFrame(best_performances, columns=['precision', 'recall', 'F-measure', 'base size'])
    for key, gdf in best_df.groupby('base size'):
        sortedf = gdf.sort_values(['precision', 'recall'])
        plt.gca().plot(sortedf['precision'], sortedf['recall'], color=colors[key], label=int(key))

    if label == 'random':
        plt.gca().legend()
    if label == 'random':
        plt.gca().set_xlim(0.75, 1.0)
        plt.gca().set_ylim(0.1, 0.85)
    else:
        plt.gca().set_xlim(0.8, 1.0)
        plt.gca().set_ylim(0.3, 0.95)
    plt.gca().set_xlabel('precision')
    plt.gca().set_ylabel('recall')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{working_dir}{folder}/settings_phase_II.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close('all')


# @logged
def check_splits(folder, master_df: pd.DataFrame):
    # for sepset_rule in ['first', 'minimal', 'full']:
    for sepset_rule in ['minimal']:
        paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
        sns.set(style='white', font_scale=1.2, context='paper', rc=paper_rc)

        col_fail_noncollider, col_noncollider, col_fail_collider, col_collider = sns.color_palette('Paired', 4)
        col_weak = sns.color_palette("Greys", 6)[2]
        col_weak2 = sns.color_palette("Greys", 6)[3]
        col_rest = sns.color_palette("Greys", 6)[-1]

        fig, axes = plt.subplots(2, 2, figsize=(6, 5))
        axes[0][0].set_title('RBO')
        axes[0][1].set_title('non-RBO')
        for col, sublabel in enumerate(['RBO', 'post-RBO']):
            working_df = fixed(master_df, {f'detect rbo': True, 'detect post rbo': True, 'sepset rule': sepset_rule})  # first?
            working_df['weak'] = working_df[f'{sublabel} violation']
            working_df['correct non-collider'] = working_df[f'{sublabel} correct violation-collider']
            working_df['wrong non-collider'] = working_df[f'{sublabel} violation-collider'] - working_df[f'{sublabel} correct violation-collider']
            working_df['correct collider'] = working_df[f'{sublabel} correct violation-non-collider']
            working_df['wrong collider'] = working_df[f'{sublabel} violation-non-collider'] - working_df[f'{sublabel} correct violation-non-collider']
            working_df['normal test case'] = working_df[f'{sublabel} case'] - working_df['weak'] - working_df[f'{sublabel} violation-non-collider'] - working_df[f'{sublabel} violation-collider']
            working_df = working_df.groupby(['base size']).mean().reset_index()
            working_df = pd.melt(working_df, id_vars=['base size'],
                                 value_vars=['weak', 'correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider', 'normal test case'],
                                 var_name='detection type',
                                 value_name='counts')

            df = working_df
            colors = [col_weak, col_noncollider, col_fail_noncollider, col_collider, col_fail_collider, col_rest]
            ax = axes[0][col]

            margin_bottom = np.zeros(len(working_df['base size'].unique()))
            for num, detection_type in enumerate(['weak', 'correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider', 'normal test case']):
                values = list(df[df['detection type'] == detection_type].loc[:, 'counts'])

                df[df['detection type'] == detection_type].plot.bar(x='base size', y='counts', ax=ax, stacked=True,
                                                                    bottom=margin_bottom, color=colors[num], label=detection_type)
                margin_bottom += values

            if col == 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                ax.legend().remove()

            working_df = fixed(master_df, {f'detect rbo': True, 'detect post rbo': True, 'sepset rule': sepset_rule})  # first?
            working_df['correct non-collider'] = working_df[f'{sublabel} correct non-collider']
            working_df['wrong non-collider'] = working_df[f'{sublabel} non-collider'] - working_df[f'{sublabel} correct non-collider']
            working_df['correct collider'] = working_df[f'{sublabel} correct collider']
            working_df['wrong collider'] = working_df[f'{sublabel} collider'] - working_df[f'{sublabel} correct collider']
            working_df['correct collider-fail'] = working_df[f'{sublabel} correct collider-fail']
            working_df['wrong collider-fail'] = working_df[f'{sublabel} collider-fail'] - working_df[f'{sublabel} correct collider-fail']
            working_df = working_df.groupby(['base size']).mean().reset_index()
            working_df = pd.melt(working_df, id_vars=['base size'],
                                 value_vars=['correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider', 'correct collider-fail', 'wrong collider-fail'],
                                 var_name='detection type',
                                 value_name='counts')

            df = working_df
            colors = [col_noncollider, col_fail_noncollider, col_collider, col_fail_collider, col_weak2, col_weak]
            ax = axes[1][col]
            margin_bottom = np.zeros(len(working_df['base size'].unique()))
            for num, detection_type in enumerate(['correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider', 'correct collider-fail', 'wrong collider-fail']):
                values = list(df[df['detection type'] == detection_type].loc[:, 'counts'])

                df[df['detection type'] == detection_type].plot.bar(x='base size', y='counts', ax=ax, stacked=True,
                                                                    bottom=margin_bottom, color=colors[num], label=detection_type)
                margin_bottom += values

            if col == 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                ax.legend().remove()

        _, max1 = axes[0][0].get_ylim()
        _, max2 = axes[0][1].get_ylim()
        axes[0][0].set_ylim(0, 9)  # max(max1, max2))
        axes[0][1].set_ylim(0, 9)  # max(max1, max2))

        _, max1 = axes[1][0].get_ylim()
        _, max2 = axes[1][1].get_ylim()
        axes[1][0].set_ylim(0, 9)  # max(max1, max2))
        axes[1][1].set_ylim(0, 9)  # max(max1, max2))

        axes[0][0].set_xticks([])
        axes[0][0].set_xticklabels([])
        axes[0][0].set_xlabel('')
        axes[0][1].set_xticks([])
        axes[0][1].set_xticklabels([])
        axes[0][1].set_xlabel('')

        axes[0][1].set_yticks([])
        axes[0][1].set_yticklabels([])
        axes[1][1].set_yticks([])
        axes[1][1].set_yticklabels([])

        axes[0][0].set_ylabel('detection cases')
        axes[1][0].set_ylabel('normal tests')

        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{working_dir}{folder}/2-by-2-rbo-post-rbo-statistics_{sepset_rule}.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
        plt.close('all')


# @logged
def check_splits2(folder, master_df: pd.DataFrame):
    for sepset_rule in ['minimal']:
        paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
        sns.set(style='white', font_scale=1.2, context='paper', rc=paper_rc)

        col_fail_noncollider, col_noncollider, col_fail_collider, col_collider = sns.color_palette('Paired', 4)
        col_weak = sns.color_palette("Greys", 6)[2]
        col_weak2 = sns.color_palette("Greys", 6)[3]

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].set_title('RBO')
        axes[1].set_title('non-RBO')
        for col, sublabel in enumerate(['RBO', 'post-RBO']):

            working_df = fixed(master_df, {f'detect rbo': False, 'detect post rbo': False, 'sepset rule': sepset_rule})  # first?
            working_df['correct non-collider'] = working_df[f'{sublabel} correct non-collider']
            working_df['wrong non-collider'] = working_df[f'{sublabel} non-collider'] - working_df[f'{sublabel} correct non-collider']
            working_df['correct collider'] = working_df[f'{sublabel} correct collider']
            working_df['wrong collider'] = working_df[f'{sublabel} collider'] - working_df[f'{sublabel} correct collider']
            working_df['correct collider-fail'] = working_df[f'{sublabel} correct collider-fail']
            working_df['wrong collider-fail'] = working_df[f'{sublabel} collider-fail'] - working_df[f'{sublabel} correct collider-fail']
            working_df = working_df.groupby(['base size']).mean().reset_index()
            working_df = pd.melt(working_df, id_vars=['base size'],
                                 value_vars=['correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider', 'correct collider-fail', 'wrong collider-fail'],
                                 var_name='detection type',
                                 value_name='counts')

            df = working_df
            colors = [col_noncollider, col_fail_noncollider, col_collider, col_fail_collider, col_weak2, col_weak]
            ax = axes[col]
            margin_bottom = np.zeros(len(working_df['base size'].unique()))
            for num, detection_type in enumerate(['correct non-collider', 'wrong non-collider', 'correct collider', 'wrong collider']):
                values = list(df[df['detection type'] == detection_type].loc[:, 'counts'])

                df[df['detection type'] == detection_type].plot.bar(x='base size', y='counts', ax=ax, stacked=True,
                                                                    bottom=margin_bottom, color=colors[num], label=detection_type)
                margin_bottom += values

            if col == 1:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                ax.legend().remove()

        _, max1 = axes[0].get_ylim()
        _, max2 = axes[1].get_ylim()
        axes[0].set_ylim(0, 9)  # max(max1, max2))
        axes[1].set_ylim(0, 9)  # max(max1, max2))

        axes[1].set_yticks([])
        axes[1].set_yticklabels([])

        axes[0].set_ylabel('normal tests')

        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{working_dir}{folder}/1-by-2-rbo-post-rbo-statistics-no-detect_{sepset_rule}.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
        plt.close('all')


def read_master_dfs(label: str, folder=None):
    """ Read csv files into DataFrame for given data type both phase I and II. """
    master_dfs = dict()
    if folder is None:
        folder = label

    master_dfs[2] = pd.read_csv(f'{working_dir}{folder}/phase_2.csv', header=None)
    master_dfs[2].columns = list(phase_2_columns)
    del master_dfs[2]['dummy0']
    del master_dfs[2]['dummy1']
    del master_dfs[2]['dummy2']

    return master_dfs


def main(label, folder=None):
    if folder is None:
        folder = label

    master_dfs = read_master_dfs(label, folder)

    draw_phase_II(label, folder, master_dfs[2])


if __name__ == '__main__':
    rc('text', usetex=True)
    pl.rcParams['text.latex.preamble'] = [
        r'\usepackage{tgheros}',  # helvetica font
        r'\usepackage{sansmath}',  # math-font matching  helvetica
        r'\sansmath'  # actually tell tex to use it!
        r'\usepackage{siunitx}',  # micro symbols
        r'\sisetup{detect-all}',  # force siunitx to use the fonts
    ]

    pd.set_option('expand_frame_repr', False)
    main('random')  # Phase-I, II valid.
