import matplotlib.pyplot as plt
import matplotlib_venn
import numpy as np
import pandas as pd

##############################Different thresholds##############################
bac_predictions = pd.read_csv(
    "code_modules/nn_training/application/small_genes_results/small_genes_results.csv",
    index_col=0)

df = pd.read_csv("data/sberro_small_genes/sberro_small_genes.csv",
                 index_col=0)

df['is_bac'] = df.index.isin(bac_predictions.index)
df = df.merge(bac_predictions, how='outer', left_index=True,
              right_index=True)

df.rename(columns={'Is_family_predicted_to_be_antimicrobial': "is_am",
                   'Confidence_score_in_antimicrobial_peptide': "am_score",
                   'probability_bacteriocin': "bac_score"},
          inplace=True)
df = df[["is_am", "am_score", "is_bac", "bac_score"]]


def get_index_set(series):
    return set(np.where(series)[0])


def make_venn(thr1, thr2, in_ax=None, show_thresholds=True):
    if in_ax is None:
        f, in_ax = plt.subplots()
    df['is_am'] = df.am_score >= thr2
    df['is_bac'] = df['bac_score'] >= thr1

    matplotlib_venn.venn3(
        subsets=[get_index_set(df.is_bac), get_index_set(df.is_am),
                 get_index_set(np.ones_like(df.is_bac))],
        set_labels=['CNNPAR', 'AmPEP', 'All'], ax=in_ax)
    if show_thresholds:
        in_ax.set_title(f"Me {thr1}, them {thr2}")


plt.close('all')
fig, axes = plt.subplots(2, 3, figsize=[12, 8])
axes = axes.flatten()
threshold = 0.99
for i, threshold_them in enumerate([0.5, 0.53, 0.58, 0.63, 0.86]):
    ax = axes[i]
    make_venn(thr1=threshold, thr2=threshold_them, in_ax=ax)

try:
    axes[i + 1].remove()
except IndexError:
    pass

plt.show()

#%%

make_venn(0.99, .50, show_thresholds=False)
plt.savefig("paper/figures/small_gene_venn_diagram")
# plt.show()

#%%

plt.close('all')
f, in_ax = plt.subplots(figsize=[8, 6])

f: plt.Figure
f.get_size_inches()
df['is_am'] = df.am_score >= 0.5
df['is_bac'] = df['bac_score'] >= 0.99

matplotlib_venn.venn3(
    subsets=[get_index_set(df.is_bac), get_index_set(df.is_am),
             set(np.where(df.bac_score == 1)[0])],
    set_labels=['CNNPAR >99%', 'AmPEP', 'CNNPAR 100%'], ax=in_ax, )

# Add percent to labels
[t.set_text(f"{t.get_text()} / {int(t.get_text()) / len(df) * 100:.2f}%")
 for t in in_ax.texts if t.get_text().isnumeric()]

in_ax: plt.Axes
# in_ax.set_title(f"Total peptides: {len(df)}")
in_ax.text(-0.2, 0.55, f"Total peptides: {len(df)}")

plt.tight_layout()
plt.savefig("paper/figures/small_gene_venn_diagram2")
