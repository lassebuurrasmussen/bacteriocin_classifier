import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def show_rbs_dist(in_df, certainty_threshold=0.9,
                  column='Fraction of family members with RBS motif',
                  get_nan=False, get_baseline=False, prob_col="probability_bacteriocin",
                  force_color=None, force_label=None, dashed_line=False):
    cmap = plt.get_cmap('Blues')
    if get_baseline:
        plot_df = in_df[column]
    elif get_nan:
        plot_df = in_df.loc[
            in_df[prob_col].isna(), column]
    else:
        plot_df = in_df.loc[in_df[prob_col] >= certainty_threshold, column]
    if not len(plot_df):
        return

    if get_baseline:
        color = 'red'
        label = 'baseline'
    elif get_nan:
        color = 'red'
        label = "CNNPAR not bacteriocin"
    else:
        color = cmap(certainty_threshold + 0.1)
        label = f">= {int(certainty_threshold * 100)}% certainty"

    if force_color is not None:
        color = force_color
    if force_label is not None:
        label = force_label

    kwargs = dict(a=plot_df, hist=False, label=label, color=color)

    if dashed_line:
        kwargs['kde_kws'] = {'linestyle': 'dashed'}

    sns.distplot(**kwargs)


if __name__ == '__main__':
    df = pd.read_excel("data/sberro_small_genes/1-s2.0-S0092867419307810-mmc5.xlsx", sheet_name=1)
    rbs_df = df[['Small protein family ID', 'Fraction of family members with RBS motif',
                 'Is family predicted to be antimicrobial']]
    df2 = pd.read_csv("code_modules/nn_training/application/small_genes_results"
                      "/small_genes_results.csv", index_col=0)
    merge_df = pd.merge(rbs_df, df2, 'outer', left_index=True, right_index=True)

    plt.close('all')
    fig, ax = plt.subplots(figsize=[10, 7])
    [show_rbs_dist(in_df=merge_df, certainty_threshold=c) for c in np.linspace(0, 1, 5)]
    show_rbs_dist(in_df=merge_df, get_nan=True, dashed_line=True)
    show_rbs_dist(in_df=merge_df, get_baseline=True)
    show_rbs_dist(in_df=merge_df, prob_col='Is family predicted to be antimicrobial',
                  certainty_threshold=1, force_color='green', force_label='AmPEP', dashed_line=True)
    ax.set_ylabel('Density')
    ax.set_title("Distributions of RBS motif fraction")
    plt.tight_layout()
    plt.savefig("./presentations/defense/plots/RBS_motif_fraction")
