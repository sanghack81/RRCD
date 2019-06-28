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


def read_master_dfs(label: str, folder=None):
    """ Read csv files into DataFrame for given data type both phase I and II. """
    master_dfs = dict()
    if folder is None:
        folder = label

    master_dfs[2] = pd.read_csv(f'{working_dir}{folder}/naive_CUT_phase_2.csv', header=None)
    master_dfs[2].columns = list(phase_2_columns)
    del master_dfs[2]['dummy0']
    del master_dfs[2]['dummy1']
    del master_dfs[2]['dummy2']

    return master_dfs


def draw_phase_II(label: str, folder, master_df: pd.DataFrame):
    master_df['precision'] = float('nan')
    safe = master_df['directed'] != 0
    master_df.loc[safe, 'precision'] = master_df.loc[safe, 'correct directed'] / master_df.loc[safe, 'directed']
    master_df['recall'] = master_df['correct directed'] / master_df['directables']
    master_df['F-measure'] = 2 * master_df['precision'] * master_df['recall'] / (master_df['precision'] + master_df['recall'])

    perform_check_best_settings(label, folder, master_df)


# @logged
@reproducible
def perform_check_best_settings(label: str, folder, master_df: pd.DataFrame, verbose=True):
    # not to choose base size
    fixables = sorted(['base size'])

    for col in fixables:
        _print_(col, verbose=verbose)
        for val in sorted(master_df[col].unique()):
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
    _print_('precision', verbose=verbose)
    _print_(newdf.sort_values('precision', ascending=False).iloc[:10], verbose=verbose)
    _print_('recall', verbose=verbose)
    _print_(newdf.sort_values('recall', ascending=False).iloc[:10], verbose=verbose)
    _print_('F-measure', verbose=verbose)
    _print_(newdf.sort_values('F-measure', ascending=False).iloc[:10], verbose=verbose)




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
