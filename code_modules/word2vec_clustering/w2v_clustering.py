#%%
import importlib

from numpy.ma import corrcoef

import code_modules.encoding.aa_encoding_functions as enc
import code_modules.word2vec_clustering.functions as fncs

importlib.reload(fncs)
importlib.reload(enc)
#%%
importlib.reload(fncs)
importlib.reload(enc)
data = fncs.load_and_combine_data()
data['length'] = data['Sequence'].apply(len)

# Embed sequences
data_seq = enc.expand_seqstr_series(data['Sequence'])
kmers_encoded = enc.w2v_embedding_encode(data_seq)

#%% Make PCA
importlib.reload(fncs)
importlib.reload(enc)
X, explained_variance = fncs.do_pca(kmers_encoded, 2, copy=False)

color_dict = fncs.make_color_dict(data)

[corrcoef(X[:, i], data['length'])[0, 1].round(3)
 for i in [0, 1]]

fncs.make_plot(xyarray=X, bacteriocin_series=data['type'],
               pc_vars=explained_variance, custom_color_dict=color_dict,
               title=f'Blank', alpha=1, save_plot=True,
               save_path="plots/sliding_window_plots/PCA_full_sequence",
               show_plot=False)

#%% Run PLS
importlib.reload(fncs)
importlib.reload(enc)

y_full = data['type'].map({'bactibase/bagel': 1, 'bacteriocin_CAMP': 1}).fillna(0)

kmers_encoded_tr = fncs.do_pls(kmers_encoded, y_full)

[corrcoef(column, data['length'])[0, 1].round(3)
 for column in kmers_encoded_tr.T]

fncs.make_plot(xyarray=kmers_encoded_tr,
               bacteriocin_series=y_full.map({1: 'bacteriocin',
                                              0: 'not bacteriocin'}),
               pc_vars=None, custom_color_dict={'not bacteriocin': 'red',
                                                'bacteriocin': 'blue'},
               title=f'Blank', s=3, alpha=0.5, save_plot=True,
               save_path="plots/sliding_window_plots/PLS_full_sequence",
               show_plot=False, figsize=[12, 7])
