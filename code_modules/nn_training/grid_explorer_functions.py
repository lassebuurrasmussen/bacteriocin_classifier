import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def extract_grid_results(path_str=None):
    if path_str is None:
        path_str = ('code_modules/nn_training/BAC_UNI_len2006/gridsearch_log_'
                    'RNNCNN31jul.txt')

    with open(path_str, 'r') as f_:
        results = f_.read().splitlines()

    results = [ast.literal_eval(r) for r in results]

    loss_acc = pd.DataFrame(
        np.array([r['result'] for r in results]).reshape(len(results), 4),
        columns=['val_error', 'val_acc', 'train_error', 'train_acc'])
    for r in results:
        r.update(r['in_kwargs'])
        del r['in_kwargs']
        del r['result']

    return pd.concat([pd.DataFrame(results), loss_acc], axis=1).drop(
        columns=['p_i'])


def plot_correlations(result_df):
    f, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    corr = result_df.corr()
    sorted_idx = corr.abs().sort_values('val_acc').index[::-1]
    corr = corr.loc[sorted_idx, sorted_idx]
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    plt.show()


def boxplot(x, y, hue, col, row, data, kind='box', min_y=None, max_y=None):
    catplot = sns.catplot(x=x, y=y, data=data,
                          hue=hue, col=col, row=row,
                          kind=kind, legend_out=False, margin_titles=True)
    [ax.grid() for ax in catplot.axes.flatten()]

    if min_y is not None:
        [ax.set_ylim(min_y) for ax in catplot.axes.flatten()]

    if max_y is not None:
        [ax.set_ylim(top=max_y) for ax in catplot.axes.flatten()]

    plt.show()
