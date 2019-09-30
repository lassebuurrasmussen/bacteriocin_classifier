# %%
import code_modules.encoding.aa_encoding_functions as enc
from code_modules.data_preprocessing.process_shortdnaDB import data

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# %% Get data and encode kmers
data_seq = enc.expand_seqstr_series(data['Sequence'])

kmers_encoded = enc.w2v_embedding_encode(data_seq)

# %% Sum on each of the 200 dimensions and normalize by kmer count
kmers_encoded_summed = kmers_encoded.reshape(len(kmers_encoded), -1, 200).sum(1)
kmer_counts = data['Sequence'].apply(len) - 2
kmers_encoded_normalized = kmers_encoded_summed / kmer_counts[:, None]


# %% PCA of summed embeddings

def do_pca(in_array, n_components):
    pca = PCA(n_components=n_components)
    fitted_data = pca.fit_transform(in_array)
    return fitted_data, pca.explained_variance_ratio_


X, explained_variance = do_pca(kmers_encoded_normalized, 2)

# %%
standard_colors = [[0.267004, 0.004874, 0.329415, 1.],
                   [0.127568, 0.566949, 0.550556, 1.],
                   [0.993248, 0.906157, 0.143936, 1.]]


def plot_pca_result(in_x, title, in_explained_variance, in_labels=(0,)):
    plt.figure(figsize=[16, 9])

    for label in pd.unique(in_labels):
        dat = (in_x[:, :2] if len(in_labels) == 1 else
               in_x[:, :2][in_labels == label])
        color = ('#1f77b4' if len(in_labels) == 1 else
                 standard_colors[label])
        plt.plot(*dat.T, '.', markersize=1,
                 color=color)
        plt.gca().set_title(title)

    plt.xlabel(f'PC1({in_explained_variance[0] * 100:.2f}% of var)')
    plt.ylabel(f'PC2({in_explained_variance[1] * 100:.2f}% of var)')

    plt.show()


plot_pca_result(X, 'PCA summed embeddings', explained_variance)

# %% PCA of unsummed
X_unsummed, explained_variance_unsummed = do_pca(kmers_encoded, 3)

# %%
plot_pca_result(X_unsummed, 'PCA UNsummed embeddings',
                explained_variance_unsummed)

# %% Kmeans of unsummed

kmeans = KMeans(n_clusters=3)
kmeans.fit(kmers_encoded)

# pd.DataFrame(np.c_[data['ID'], kmeans.labels_], columns=[
#     'ID', 'cluster']).rename_axis('index').to_csv(
#     "data/shortCompleteUniquegenes/shortdnaDB_w2v_clustering.csv")

# %% Plot it with colors
plot_pca_result(X_unsummed, 'PCA UNsummed embeddings',
                explained_variance_unsummed, in_labels=kmeans.labels_)

# %% Make kernel density plot
sns.jointplot(*X_unsummed[:, :2].T, kind='kde')
plt.show()

# %%


f, ax = plt.subplots(figsize=[16, 9], subplot_kw=dict(projection='3d'))

ax.scatter(*X_unsummed.T, s=1, c=kmeans.labels_)
[getattr(ax, f'set_{d}label')(
    f'PC{l + 1}({explained_variance_unsummed[l] * 100:.2f}% of var) of var')
    for l, d in enumerate(list('xyz'))]
ax.set_title('PCA UNsummed embeddings')
plt.show()

# %%

X_unsummed2d = X_unsummed[:, :2]

df = pd.DataFrame(np.c_[X_unsummed2d, kmeans.labels_],
                  columns=['PC1', 'PC2', 'label'])

df[['PC1_groupmean', 'PC2_groupmean']] = df.groupby('label').transform(lambda grp: grp.mean())

diff_vec = df[['PC1', 'PC2']].values - df[['PC1_groupmean', 'PC2_groupmean']].values
df['dist_to_mean'] = np.apply_along_axis(np.linalg.norm, 1, diff_vec)

representers = df.groupby('label').apply(lambda grp: grp.sort_values(by='dist_to_mean')[:5]).droplevel(0)

plt.plot(*representers[['PC1', 'PC2']].values.T, '.')
plt.show()

representers = pd.merge(representers[['label']], data['Sequence'], left_index=True, right_index=True)

(representers.iloc[:, 0].astype(str) + '-' + representers.iloc[:, 1]).tolist()

# %%

[pd.Series([seq[i:i + 3] for i in range(len(seq)) if len(seq[i:i + 3]) == 3]).value_counts()
 for seq in representers['Sequence'].tolist()]
