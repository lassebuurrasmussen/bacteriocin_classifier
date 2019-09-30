import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from matplotlib.legend import Legend
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

word2vec_model = Word2Vec.load("data/hamid_wordvec_model_trembl_size_200")

# Get entire vocabulary from Word2Vector model
vocab = np.array(list(word2vec_model.wv.vocab))

# Get word vector for each vocabulary entry
vws = pd.DataFrame(
    np.apply_along_axis(lambda word: word2vec_model.wv.get_vector(word[0]),
                        axis=1, arr=vocab.reshape(-1, 1)), index=vocab)

# %% Dimensionality reduction
pca = PCA(n_components=2)
vws_2d = pd.DataFrame(pca.fit_transform(vws),
                      index=vws.index, columns=['PC1', 'PC2'])


# %% Plot all words
def make_text_plot(in_2dvocab, text_col='index'):
    plt.close('all')
    plt.figure(figsize=[32, 32])
    plt.axis(np.c_[in_2dvocab[['PC1', 'PC2']].min(),
                   in_2dvocab[['PC1', 'PC2']].max()].flatten())

    if text_col == 'index':
        for row in in_2dvocab.iloc[:].iterrows():
            plt.text(row[1]['PC1'], row[1]['PC2'], row[0],
                     fontdict={'fontsize': 6})
    else:
        for row in in_2dvocab.iloc[:].iterrows():
            plt.text(row[1]['PC1'], row[1]['PC2'], row[1][text_col],
                     fontdict={'fontsize': 6})

    plt.show()


make_text_plot(vws_2d)

# Seems like there are three main clusters, where two of them contain a lot of Xs

# %% Plot by whether word contains X

plt.figure()
plt.scatter(x=vws_2d['PC1'], y=vws_2d['PC2'], c=vws_2d.index.str.contains('X'),
            s=2)
plt.show()

# This confirms what we observed before

# %% Perform PCA again and remake the plot without Xs
pca_noXs = PCA(n_components=2)
vws_noXs = vws[~vws.index.str.contains('X')]
vws_noXs_2d = pd.DataFrame(pca_noXs.fit_transform(vws_noXs),
                           index=vws_noXs.index,
                           columns=['PC1', 'PC2'])

make_text_plot(vws_noXs_2d)

# Again three major clusters. This time it seems like the cysteins are what
# makes the difference

# %% Cluster the amino acids and plot it

kmeans = KMeans(n_clusters=3)
kmeans.fit(vws_noXs)


#%%


def plot_df(in_df, color_series, ax=None, colormap='cool', in_title=''):
    # in_df.plot(kind='scatter', x='PC1', y='PC2', c=color_series,
    #            colormap=colormap, alpha=0.7, ax=ax)
    in_df.plot(kind='scatter', x='PC1', y='PC2', c=color_series, alpha=0.7,
               ax=ax)
    if not ax:
        plt.show()

    else:
        ax.set_title(in_title)


#%%

plot_df(vws_noXs_2d, kmeans.labels_)

# %% Let's look at the C content of each cluster

vws_clustered = vws_noXs_2d.rename_axis(index='word').reset_index()
vws_clustered['cluster'] = kmeans.labels_

vws_clustered.groupby('cluster')['word'].apply(
    lambda row: row.str.contains('C').mean())

vws_noXs_2d.plot(kind='scatter', x='PC1', y='PC2',
                 c=vws_noXs_2d.index.str.contains('C'), colormap='cool',
                 alpha=0.7)
plt.show()

# Much higher fraction of C in one cluster

# %% Plot only the Cs
vws_c_only = vws_clustered[vws_clustered['word'].str.contains('C')]

make_text_plot(vws_c_only, text_col='word')

# %% Explore what the difference between Cs in one cluster and the other is
letter_fracs = vws_c_only.copy()['word'].apply(
    lambda row: pd.value_counts(list(row)) / 3).fillna(0)

letter_fracs['cluster'] = kmeans.labels_[vws_c_only.index]
letter_fracs['cluster'].replace(1, 2, inplace=True)

letter_fracs_mean = letter_fracs.groupby('cluster').mean()

letter_fracs_mean.loc[:, (letter_fracs_mean == 0).any(0)]

# The second cluster is the one containing the "strange" amino acids BUO

# %% Extract only real AAs
pca_oreal = PCA(n_components=2)

vws_oreal = vws_noXs[~vws_noXs.index.str.contains('[UOB]')].copy()

vws_oreal_2d = pd.DataFrame(pca_oreal.fit_transform(vws_oreal),
                            index=vws_oreal.index, columns=['PC1', 'PC2'])

kmeans_oreal = KMeans(n_clusters=3)
kmeans_oreal.fit(vws_oreal)

# %% Plot resulting clusters
plt.close('all')
for cluster in np.unique(kmeans_oreal.labels_):
    df = vws_oreal_2d[kmeans_oreal.labels_ == cluster]
    plt.scatter(df['PC1'], df['PC2'], label=cluster, alpha=0.7)

plt.legend()
plt.show()

# %% And plot all words
make_text_plot(vws_oreal_2d)

# %% Inspect difference in word contents

vws_oreal_2d_clustered = vws_oreal_2d.assign(
    cluster=kmeans_oreal.labels_).rename_axis(index='word').reset_index()

letter_fracs: pd.DataFrame = vws_oreal_2d_clustered.apply(
    lambda row: pd.value_counts(list(row['word'])), axis=1)
letter_fracs.fillna(0, inplace=True)
letter_fracs['cluster'] = kmeans_oreal.labels_

(letter_fracs.groupby('cluster').sum() / len(letter_fracs)).T


# Seems like C and W, Z make a big difference

# %% Visualize that difference

def containing_plot(in_df, pattern, negate_pattern=False):
    contains_bool = in_df.index.str.contains(pattern)
    contains_bool = ~contains_bool if negate_pattern else contains_bool

    plt.close('all')
    for does in [True, False]:
        label = does if not negate_pattern else not does
        df = in_df[contains_bool == does]
        plt.scatter(df['PC1'], df['PC2'], alpha=0.2,
                    label=f'Contains {pattern} - {label}')

    plt.legend()
    plt.show()


containing_plot(vws_oreal_2d, pattern='[W]')
containing_plot(vws_oreal_2d, pattern='C')
containing_plot(vws_oreal_2d, pattern='[CW]', negate_pattern=True)

#%%

sns.set()
sns.reset_defaults()


def all_letters_plot(data_frame=vws_2d, fname=''):
    letters = sorted(set(list("".join(vws_2d.index.tolist()))))
    plt.close('all')
    fig, axes = plt.subplots(5, 5, figsize=[11, 9])
    axes = axes.flatten()
    for i, letter in enumerate(letters):
        ax: plt.Axes = axes[i]

        data_frame = data_frame.copy()
        data_frame['Contains letter'] = data_frame.index.str.contains(letter)

        legend = 'brief' if i == 0 else False

        scat = sns.scatterplot(x='PC1', y='PC2',
                               hue='Contains letter', size=3, linewidth=0.2,
                               data=data_frame, ax=ax, legend=legend)

        if i == 0:
            scat: plt.Axes
            handles, labels = scat.get_legend_handles_labels()
            handles, labels = ([h for i, h in enumerate(handles) if
                                labels[i] not in ['2', '3']],
                               [l for l in labels if l not in ['2', '3']])
            scat.legend(handles=handles, labels=labels)
            leg: Legend = scat.get_legend()
            leg.set_bbox_to_anchor([0.82, 1.8])

        ax.set_title(letter)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(f'paper/figures/w2v_contains_letter{fname}.png')


all_letters_plot()

pca_noXUs = PCA(n_components=2)
vws_noXUs = vws[~vws.index.str.contains('[XU]')]
vws_noXUs_2d = pd.DataFrame(pca_noXs.fit_transform(vws_noXUs),
                            index=vws_noXUs.index,
                            columns=['PC1', 'PC2'])

all_letters_plot(data_frame=vws_noXUs_2d, fname="_noXU")
