import warnings
from itertools import product

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc, gridspec

from rrpcd._UAI_2019.unified_analyze_utils import logged, _draw_general
from rrpcd.experiments.exp_utils import fixed

warnings.filterwarnings("error")
working_dir = f'rrpcd/_UAI_2019/'

phase_1_id_vars = ('idx', 'base size', 'aggregation', 'order-dependence')
phase_1_columns = (
    *phase_1_id_vars,
    'true positives', 'false positives',
    'dependencies',
    'right direction (aggregated)', 'reverse direction (aggregated)', 'false positives (aggregated)'
)


def draw_phase_I(folder, master_df: pd.DataFrame):
    """ Macro average """
    if True:
        draw_phase_1_macro_average(folder, fixed(master_df, {'order-dependence': False, 'aggregation': True}), sns.color_palette("Paired", 6))

    if True:
        draw_order_independence(folder, master_df)

    if True:
        how_aggregation(folder, fixed(master_df, {'order-dependence': False, 'aggregation': True}))
        draw_aggregation(folder, master_df)


@logged
def draw_order_independence(folder, master_df: pd.DataFrame):
    columns_to_mean = ['true positives', 'false positives']
    original = pd.options.display.float_format
    pd.options.display.float_format = '{:.3f}'.format
    print(master_df[np.logical_or(master_df['base size'] == 200, master_df['base size'] == 500)].
          groupby(['base size', 'aggregation', 'order-dependence'])[columns_to_mean].
          mean())
    print(master_df.groupby(['aggregation', 'order-dependence'])[columns_to_mean].mean())
    pd.options.display.float_format = original

    df = master_df
    df['order-independence'] = np.logical_not(df['order-dependence'])

    df = df.melt(id_vars=['idx', 'base size', 'aggregation', 'order-independence'],
                 value_vars=['true positives', 'false positives'],
                 var_name='category',
                 value_name='counts'
                 )
    row_order = ['true positives', 'false positives']

    def hook(g):
        g.axes[0][0].set_ylim(0, 7)
        g.axes[0][1].set_ylim(0, 0.5)

    for agg_setting in [True, False]:
        if agg_setting is not None:
            work_df = df[df['aggregation'] == agg_setting].copy()
        else:
            work_df = df.copy()
        work_df['order-independence'] = work_df['order-independence'].map(lambda x: 'order-independent' if x else 'order-dependent')
        _draw_general(work_df,
                      x='base size', y='counts', hue='order-independence', data=work_df, col='category',
                      hue_order=['order-independent', 'order-dependent'], col_order=row_order,
                      palette=sns.color_palette('Set1', 2), ci=95, n_boot=2000, sharey=False,
                      aspect=1.3, height=2.3, filename=f'{working_dir}{folder}/phase_I_order_independence_agg_{agg_setting}.pdf', legend=False, legend_out=False,
                      legend_locs=(0, 1, 1), factor_plot_hook=[hook]
                      )


@logged
def draw_aggregation(folder, master_df: pd.DataFrame):
    columns_to_mean = ['true positives', 'false positives']
    original = pd.options.display.float_format
    pd.options.display.float_format = '{:.3f}'.format
    print(master_df[np.logical_or(master_df['base size'] == 200, master_df['base size'] == 500)].
          groupby(['base size', 'order-dependence', 'aggregation'])[columns_to_mean].
          mean())
    print(master_df.groupby(['order-dependence', 'aggregation'])[columns_to_mean].mean())
    pd.options.display.float_format = original
    df = master_df
    df['order-independence'] = np.logical_not(df['order-dependence'])

    df = df.melt(id_vars=['idx', 'base size', 'aggregation', 'order-independence'],
                 value_vars=['true positives', 'false positives'],
                 var_name='category',
                 value_name='counts'
                 )
    row_order = ['true positives', 'false positives']

    def hook(g):
        g.axes[0][0].set_ylim(0, 7)
        g.axes[0][1].set_ylim(0, 0.5)

    for oi in [True, False]:
        work_df = df[df['order-independence'] == oi].copy()
        work_df['aggregation'] = work_df['aggregation'].map(lambda x: 'w/ aggregation' if x else 'w/o aggregation')
        _draw_general(work_df,
                      x='base size', y='counts', hue='aggregation', data=work_df, col='category',
                      hue_order=['w/ aggregation', 'w/o aggregation'], col_order=row_order,
                      palette=sns.color_palette('Set1', 2), ci=95, n_boot=2000, sharey=False,
                      aspect=1.3, height=2.3, filename=f'{working_dir}{folder}/phase_I_aggregation_with_oi_{oi}.pdf', legend=False, legend_out=False,
                      legend_locs=(0, 1, 1), factor_plot_hook=[hook]
                      )


@logged
def how_aggregation(folder, master_df: pd.DataFrame):
    df = master_df.rename(columns={'right direction (aggregated)': 'right direction',
                                   'reverse direction (aggregated)': 'reverse direction',
                                   'false positives (aggregated)': 'false positive'})
    df['counts'] = df['right direction'] + df['reverse direction'] + df['false positive']
    how_aggregated_worked = list()
    for size in sorted(df['base size'].unique()):
        subdf = fixed(df, {'base size': size})
        counts = subdf['right direction'].sum() + subdf['reverse direction'].sum() + subdf['false positive'].sum()
        how_aggregated_worked.append([size, counts, subdf['right direction'].mean(), subdf['reverse direction'].mean(), subdf['false positive'].mean()])

    agg_df = pd.DataFrame(how_aggregated_worked, columns=['base size', 'counts', 'right direction', 'reverse direction', 'false positive'])
    draw_how_aggregated_works(folder, agg_df, df)


@logged
def draw_how_aggregated_works(folder, master_df: pd.DataFrame, df2):
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
    sns.set(style='white', font_scale=1.2, context='paper', rc=paper_rc)
    sns.set_palette(sns.color_palette("Set2"))
    plt.figure(figsize=(5.5, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

    # left, right = 0, 1
    subdf = master_df.copy()  # type: pd.DataFrame

    print("=== Average counts ======================")
    print(df2.groupby('base size')[['right direction', 'reverse direction', 'false positive']].mean())
    """ Left plot """
    df = pd.melt(df2, 'base size', ['right direction', 'reverse direction', 'false positive'], var_name='types', value_name='average count')
    ax = plt.subplot(gs[0])  # type: plt.Axes
    sns.pointplot('base size', 'average count', hue='types', data=df, ax=ax, dodge=True)
    ax.set_ylim(bottom=0.0, top=0.5)
    ax.set_title('counts')
    ax.legend(title='')
    ax.set_xlabel('base size')
    ax.set_ylabel('')

    """ Right plot """
    ax = plt.subplot(gs[1])  # type: plt.axes
    cols = ['right direction', 'reverse direction', 'false positive']
    normalizer = subdf[['right direction', 'reverse direction', 'false positive']].sum(axis=1)
    for col in cols:
        subdf[col] = subdf[col] / normalizer
    stds_df = subdf.copy()  # type: pd.DataFrame
    for col in cols:
        stds_df[col] = np.sqrt((subdf[col] * (1 - subdf[col])) / subdf['counts'])
    before_melt_subdf = subdf.copy()
    subdf = pd.melt(subdf, 'base size', ['right direction', 'reverse direction', 'false positive'], var_name='types', value_name='ratios')
    stds_df = pd.melt(stds_df, 'base size', value_vars=['right direction', 'reverse direction', 'false positive'], var_name='types', value_name='error')
    subdf['error'] = 1.96 * stds_df['error']
    df = subdf
    #
    u = sorted(df['base size'].unique())
    x = np.arange(len(u))
    subx = df['types'].unique()
    bottom = None
    for i, gr in enumerate(subx):
        dfg = df[df['types'] == gr]
        if i == 0:
            ax.bar(x, dfg['ratios'].values, label=f"{gr}", yerr=dfg['error'].values)
            bottom = dfg['ratios'].values
        else:
            ax.bar(x, dfg['ratios'].values, bottom=bottom, label=f"{gr}", yerr=dfg['error'].values)
            bottom += dfg['ratios'].values
    ax.set_xlabel('base size')
    ax.yaxis.tick_right()
    ax.set_title('proportions')
    ax.set_xticks(x)
    ax.set_xticklabels(u)
    ax.legend().set_visible(False)
    ax.set_ylim(0.0, 1.1)
    print("=== Proportions ======================")
    print(before_melt_subdf)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{working_dir}{folder}/phase_I_aggregation.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


@logged
def draw_phase_1_macro_average(folder, master_df: pd.DataFrame, pal):
    df = master_df

    df[f'precision'] = df[f'true positives'] / (df[f'true positives'] + df[f'false positives'])
    df[f'recall'] = df[f'true positives'] / df['dependencies']
    df[f'F-measure'] = 2 * df[f'precision'] * df[f'recall'] / (df[f'precision'] + df[f'recall'])

    hue_order = ['precision', 'recall', 'F-measure']

    for legend in [False, True]:
        _draw_general(df, x='base size', y='ratio', hue='metric', to_melt=True, melt_id_vars=['idx', 'base size', 'aggregation'],
                      hue_order=hue_order,
                      filename=f'{working_dir}{folder}/macro_average_legend_{legend}.pdf', palette=pal,
                      legend_locs='out' if legend else None,
                      legend=False, legend_out=False, height=3, aspect=2 if legend else 1.3, dodge=True
                      )

    return df


def read_master_dfs(label: str, folder=None):
    """ Read csv files into DataFrame for given data type both phase I and II. """
    master_dfs = dict()
    if folder is None:
        folder = label

    master_dfs[1] = pd.read_csv(f'{working_dir}{folder}/phase_1.csv', header=None)
    master_dfs[1].columns = list(phase_1_columns)

    return master_dfs


def report_phase_I(master_df):
    df = None
    print("macro average")
    for agg, od in product([True, False], repeat=2):
        for base_size in [200, 300, 400, 500]:
            df = fixed(master_df, {'aggregation': agg, 'order-dependence': od, 'base size': base_size})

            df[f'precision'] = df[f'true positives'] / (df[f'true positives'] + df[f'false positives'])
            df[f'recall'] = df[f'true positives'] / df['dependencies']
            df[f'F-measure'] = 2 * df[f'precision'] * df[f'recall'] / (df[f'precision'] + df[f'recall'])

            print(f"{agg} &  {not od} & {base_size} & {df['precision'].mean() * 100:.2f}\\% & {df['recall'].mean() * 100:.2f}\\% \\\\")
    print()
    print()
    print("micro average")
    for agg, od in product([True, False], repeat=2):
        for base_size in [200, 300, 400, 500]:
            df = fixed(master_df, {'aggregation': agg, 'order-dependence': od, 'base size': base_size})

            precision = df[f'true positives'].sum() / (df[f'true positives'].sum() + df[f'false positives'].sum())
            recall = df[f'true positives'].sum() / df['dependencies'].sum()

            print(f"{agg} &  {not od} & {base_size} & {precision*100:.2f}\\% & {recall*100:.2f}\\% \\\\")

    print()
    print()
    print("average TP, FP")
    for agg, od in product([True, False], repeat=2):
        for base_size in [200, 300, 400, 500]:
            df = fixed(master_df, {'aggregation': agg, 'order-dependence': od, 'base size': base_size})

            print(f"{agg} &  {not od} & {base_size} & {df[f'true positives'].mean():.2f}\\% & {df[f'false positives'].mean():.2f}\\% \\\\")


    return df


def main(label, folder=None):
    if folder is None:
        folder = label

    master_dfs = read_master_dfs(label, folder)

    report_phase_I(master_dfs[1])
    # draw_phase_I(folder, master_dfs[1])


if __name__ == '__main__':
    rc('text', usetex=True)
    pl.rcParams['text.latex.preamble'] = [
        r'\usepackage{tgheros}',  # helvetica font
        r'\usepackage{sansmath}',  # math-font matching  helvetica
        r'\sansmath'  # actually tell tex to use it!
        r'\usepackage{siunitx}',  # micro symbols
        r'\sisetup{detect-all}',  # force siunitx to use the fonts
    ]

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    pd.set_option('expand_frame_repr', False)
    main('random', 'random')  # Phase-I, II valid.
