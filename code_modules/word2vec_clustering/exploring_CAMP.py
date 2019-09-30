import pandas as pd
import numpy as np

import code_modules.word2vec_clustering.functions as fncs

import code_modules.encoding.aa_encoding_functions as enc

# %%


save_plots = False
save_path = "plots/CAMP_w2v_clustering_plots/"
pltcount = 0

data: pd.DataFrame = fncs.get_bacteriocins()

# Encode kmers
data_seq = enc.expand_seqstr_series(data['Sequence'])

kmers_encoded = enc.w2v_embedding_encode(data_seq)

# Sum on each of the 200 dimensions and normalize by kmer count
kmers_encoded_summed = kmers_encoded.reshape(len(kmers_encoded), -1, 200).sum(1)
kmer_counts = data['Sequence'].apply(len) - 2
kmers_encoded_normalized = kmers_encoded_summed / kmer_counts[:, None]

X, explained_variance = fncs.do_pca(kmers_encoded_normalized, 2)
# %%


base_title = 'CAMP,BAGEL,BACTIBASE'
fncs.make_plot(xyarray=X, bacteriocin_series=data['bacteriocin'],
               pc_vars=explained_variance,
               title=f'{base_title} - dimensions summed')

# %% PCA of unsummed
X_unsummed, explained_variance_unsummed = fncs.do_pca(kmers_encoded, 3)

# %%

fncs.make_plot(X_unsummed[:, :2], data['bacteriocin'], explained_variance_unsummed,
               title=f'{base_title} - raw dimensions')

# %%
# Select from PC thresholds
boolar = ((X_unsummed[:, 0] < 0.6) & (X_unsummed[:, 1] < 1.8))

boolar2 = ((X_unsummed[:, 0] > 7.7) & (X_unsummed[:, 0] < 8.5) &
           (X_unsummed[:, 1] > -3.5) & (X_unsummed[:, 1] < -2.))

# Select with line
a, b = 1., 2.15
line = (a * X_unsummed[:, 0] + b - X_unsummed[:, 1]) > 0
boolar = (boolar & line) | boolar2


def in_subset_frac(bac_label):
    subset = (boolar & (data['bacteriocin'] == bac_label))
    non_subset = (~boolar & (data['bacteriocin'] == bac_label))
    return subset.sum() / (np.sum(subset) + np.sum(non_subset))


ratio_dict = {}
for bac in data['bacteriocin'].unique():
    ratio_dict[bac] = in_subset_frac(bac) * 100

plot_args2 = dict(xyarray=X_unsummed[:, :2],
                  bacteriocin_series=boolar.astype(int),
                  pc_var_array=explained_variance_unsummed,
                  custom_color_dict=pd.Series({1: 'r', 0: 'b'}))
fncs.make_plot(**plot_args2, subset_list=[ratio_dict],
               title=f'{base_title} - raw dimensions subsetted')

# %%
lens = data['Sequence'].apply(len).values

np.apply_along_axis(lambda col: np.corrcoef(col, lens)[0, 1], 0, X_unsummed)

X_len_normalized = X_unsummed[:, :2].copy()
X_len_normalized[:, 0] /= lens

var_ratio = explained_variance_unsummed[0] / X_unsummed.var(0)[0]
new_variance_explained = np.array([X_len_normalized[:, 0].var() * var_ratio,
                                   explained_variance_unsummed[1]])

fncs.make_plot(X_len_normalized, data['bacteriocin'], new_variance_explained,
               title=f'{base_title} - PC1 divided by length')

# %%
plot_args3 = dict(xyarray=X_unsummed[:, :0:-1],
                  bacteriocin_series=data['bacteriocin'],
                  pc_var_array=explained_variance_unsummed[:0:-1],
                  title=f'{base_title} - NO PC1',
                  custom_axis_labels=['PC3', 'PC2'])
fncs.make_plot(**plot_args3)
fncs.make_plot(**plot_args3, multiple_axes=True)

# %%
data['class'] = data['class'].map({0: 'bactibase', 1: 'bagelC1', 2: 'bagelC2',
                                   3: 'bagelC3', 4: 'CAMPbacteriocin'})

bool_classes = ~data['class'].isna()
fncs.make_plot(xyarray=X_unsummed[bool_classes, :0:-1],
               bacteriocin_series=data.loc[~data['class'].isna(), 'class'],
               pc_vars=explained_variance_unsummed[:0:-1],
               custom_color_dict={'bactibase': 'r', 'bagelC1': 'b', 'bagelC2': 'g',
                                  'bagelC3': 'y', 'CAMPbacteriocin': 'c'},
               title=f'{base_title} bacteriocin classes', multiple_axes=False,
               s=8, alpha=1,
               axnrowscols=[2, 3], custom_axis_labels=['PC3', 'PC2'])
