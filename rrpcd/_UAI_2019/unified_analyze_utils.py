import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from rrpcd.experiments.exp_utils import fixed


def inspect_fixed(master_df, df=None):
    if df is None:
        colval = dict()
        for col in master_df.columns:
            if len(master_df[col].unique()) == 1:
                colval[col] = master_df[col].unique()[0]
        if colval:
            print(f">> given data frame with a single value: {colval}", file=sys.stderr, flush=True)
    else:
        removed = set(master_df.columns) - set(df.columns)
        new_cols = set(df.columns) - set(master_df.columns)
        if removed:
            print(f'>> removed columns: {removed}')
        if new_cols:
            print(f'>> new columns:     {[col for col in df.columns if col in new_cols]}')
        for col in df.columns:
            if len(df[col].unique()) == 1:
                print(f">> used data frame with a single value: {col.rjust(30)}: {df[col].unique()[0]}", file=sys.stderr, flush=True)


def logged(f):
    def fff(label: str, folder, master_df: pd.DataFrame, *args, **kwargs):
        time.sleep(0.1)
        print("\n\n\n\n", file=sys.stderr, flush=True)
        print("===================================================", file=sys.stderr, flush=True)
        print(f'starting {f.__name__} for label: {label}', file=sys.stderr, flush=True)
        inspect_fixed(master_df)
        time.sleep(0.1)
        df = f(label, folder, master_df, *args, **kwargs)
        time.sleep(0.1)
        if df is not None:
            inspect_fixed(master_df, df)

        print('ending   ' + f.__name__, file=sys.stderr, flush=True)
        print("==================================================", file=sys.stderr, flush=True)
        time.sleep(0.1)

    return fff


def _print_(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)


def _header_hook(row_names=None, col_names=None, xlabel=None):
    def inner_hook(g):
        # post factorplot
        if row_names:
            for ax, row_name in zip(g.axes, row_names):
                ax[0].set(ylabel=row_name)
        if col_names:
            for ax in g.axes.flat:
                ax.set_title('')

            for ax, col_name in zip(g.axes[0], col_names):
                ax.set_title(col_name)
        if xlabel is not None:
            for ax in g.axes[-1]:
                ax.set(xlabel=xlabel)

    return inner_hook


def _draw_general(master_df: pd.DataFrame, *,
                  to_fix=None,
                  x=None,
                  y=None,
                  row=None,
                  col=None,
                  hue=None,
                  row_order=None,
                  col_order=None,
                  hue_order=None,
                  to_melt=None,
                  melt_id_vars=None,
                  figsize=(6, 4),
                  font_scale=1.2,
                  palette=None,
                  filename=None,
                  y_lim=None,
                  factor_plot_hook=None,
                  legend_locs=(-1, -1, 4), **kwargs):
    time.sleep(0.1)
    print(f'  drawing: {filename}', file=sys.stderr, flush=True)
    fixed_found = {col: master_df[col].unique()[0] for col in master_df.columns if len(master_df[col].unique()) == 1}
    if fixed_found:
        print('  Found Fixed::', fixed_found, file=sys.stderr, flush=True)
        time.sleep(0.1)

    # prepare data
    working_df = master_df
    if to_fix is not None:
        working_df = fixed(working_df, to_fix)

    if to_melt:
        working_df = pd.melt(working_df, id_vars=melt_id_vars,
                             value_vars=hue_order, var_name=hue, value_name=y)

    # prepare
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 2}
    sns.set(style='white', font_scale=font_scale, context='paper', rc=paper_rc)
    plt.figure(figsize=figsize)
    if palette is not None:
        sns.set_palette(palette)

    # prepare for options
    factor_plot_dict = {'x': x, 'y': y, 'data': working_df, 'kind': 'point'}

    if hue is not None:
        factor_plot_dict.update(hue=hue)
        if hue_order is not None:
            factor_plot_dict.update(hue_order=hue_order)

    if row is not None:
        factor_plot_dict.update(row=row)
        if row_order is not None:
            factor_plot_dict.update(row_order=row_order)

    if col is not None:
        factor_plot_dict.update(col=col)
        if col_order is not None:
            factor_plot_dict.update(col_order=col_order)
    if kwargs is not None:
        factor_plot_dict.update(**kwargs)

    g = sns.catplot(**factor_plot_dict)

    if y_lim is not None:
        for ax in g.axes.flat:
            ax.set_ylim(*y_lim)

    if legend_locs:
        if legend_locs == 'out':
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        else:
            g.axes[legend_locs[0]][legend_locs[1]].legend(loc=legend_locs[2])

    if factor_plot_hook is not None:
        if isinstance(factor_plot_hook, list):
            for hook in factor_plot_hook:
                hook(g)

    # saving and closing
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close('all')
