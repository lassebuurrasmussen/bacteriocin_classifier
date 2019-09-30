import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from code_modules.nn_training import tflogs2pandas as tf2pd

#%%
importlib.reload(tf2pd)

tf2pd.dump_to_csv(
    log_dir='code_modules/nn_training/BAC_UNI_len2006/logs',
    output_dir='code_modules/nn_training/BAC_UNI_len2006/csv_logs')

dir_path = "code_modules/nn_training/BAC_UNI_len2006/"

tf2pd.extract_results_from_csv(
    pd_df_path=dir_path, out_file_name='BAC_UNI_len2006_result.csv')

#%%
# Load training results
df = pd.read_csv('code_modules/nn_training/BAC_UNI_len2006/csv_logs/'
                 'all_training_logs_in_one_file.csv')
# Load best epochs
best_epochs = pd.read_csv("code_modules/nn_training/BAC_UNI_len2006/BAC_UNI_"
                          "len2006_result.csv",
                          index_col=0).groupby(['embedding',
                                                'architecture'])['best_epoch']

# Assert that mean is based on unique numbers
assert all(best_epochs.nunique() == 1)

# Discard all other epochs
df = df[df['step'] == df.merge(best_epochs.mean().reset_index(), 'left',
                               ['embedding', 'architecture'])['best_epoch']]


def catplot(in_df, save_fig=False, out_fname=None, x='architecture',
            order=None, col=None, hue='run_type'):
    plt.close('all')

    in_df = in_df.copy()
    in_df['embedding'] = in_df['embedding'].replace("elmo", 'ELMo').replace(
        "w2v", 'Word2Vec')

    p = sns.catplot(x=x, y='value', hue=hue,
                    ci='sd',
                    order=order,
                    kind='point',
                    data=in_df,
                    legend_out=False, height=7,
                    capsize=0.1, col=col)
    [ax.grid() for ax in p.axes.flatten()]
    ax: plt.Axes

    # breakpoint()
    xtick_mapper = {'RNNCNN': 0, 'DNN': 1, 'CNNPAR': 2, 'CNN': 3,
                    'CNNPARLSTM': 4, 'BIDGRU': 5}

    # {k:v - 1 for k, v in xtick_mapper.items()}

    label_df = (in_df.groupby(['architecture', 'embedding'])['value'].mean()
                .reset_index())

    [[ax.text(xtick_mapper[x], y_, f"{s:.3f}") for x, y_, s
      in zip(label_df['architecture'], label_df['value'], label_df['value'])]
     for ax in p.axes.flatten()]

    [ax.set_yticks(np.arange(round(ax.get_ylim()[0], 2),
                             round(ax.get_ylim()[1], 2) + 0.01, 0.01))
     for ax in p.axes.flatten()]

    [[ax.set_xlabel('Architecture'), ax.set_ylabel('Accuracy')] for ax in
     p.axes.flatten()]

    if save_fig:
        # path = "code_modules/nn_training/BAC_UNI_len2006/"
        path = "paper/figures/"

        if out_fname is None:
            out_fname = "BAC_UNI_len2006_elmo_result"

        plt.tight_layout()
        plt.savefig(path + out_fname)

    else:
        plt.show()


catplot(df.query("metric == 'epoch_acc' & run_type == 'validation'"),
        x='architecture', order=None, hue='embedding',
        out_fname="BAC_UNI_len2006_elmo_vs_w2v_result", save_fig=True)

#%%
##########################Dists not equal var###################################
plt.close('all')
df2 = df.query("metric == 'epoch_acc' & run_type == 'validation'").drop(
    ['metric', 'run_type', 'step'], axis=1)

g = sns.FacetGrid(data=df2, row='embedding', col='architecture')

g.map(sns.kdeplot, 'value')
plt.show()

####################################P Values####################################
pvals = {}
for architecture in df2['architecture'].unique():
    w2v = df2.query("embedding == 'w2v' & architecture == @architecture")[
        'value'].values
    elmo = df2.query("embedding == 'elmo' & architecture == @architecture")[
        'value'].values

    # Welch’s t-test  ################
    pvals[architecture] = stats.ttest_ind(w2v, elmo, equal_var=False).pvalue

print(str({k: f"{v:.3g}" for k, v in pvals.items()}).translate({ord("{"): '',
                                                                ord("}"): '',
                                                                ord("'"): ''}))

print(" ".join([f"{v:.3g}" for v in pvals.values()]))
