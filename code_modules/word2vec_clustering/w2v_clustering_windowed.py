#%%
# # To run with nohup
# import os
#
# os.chdir('pycharm_project/Master Thesis')
import importlib

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import code_modules.encoding.aa_encoding_functions as enc
import code_modules.word2vec_clustering.functions as fncs

importlib.reload(fncs)
importlib.reload(enc)

data = fncs.load_and_combine_data()
data['length'] = data['Sequence'].apply(len)

# # Subset data for testing
# idxs = np.concatenate([np.random.choice(data[data['type'] == t].index, 10,
#                                         False)
#                        for t in data['type'].unique()])
# data = data.loc[idxs].reset_index(drop=True)


DO_PLOTTING = False
if DO_PLOTTING:
    fncs.differing_window_plots(data)

X, y = enc.w2v_embedding_window_encode(data, data['Sequence'], w=50)

y_binary = (y.map({'bactibase/bagel': 1, 'bacteriocin_CAMP': 1}).fillna(0).
            map({1: 'bacteriocin', 0: 'not bacteriocin'}))

# PLS
y_binary_int = y_binary.map({'bacteriocin': 1, 'not bacteriocin': 0})
# Xs_pls, reg = fncs.do_pls(X, y_binary_int, get_estimator=True)


ranking = fncs.load_ranking()

pipe = Pipeline(steps=[('col_selection', fncs.MyColPicker()),
                       ('classifier', fncs.MyPLSClassifier())])

col_ns = [3, 10, 30, 60, 100, 200, 300, 500, 1000, 2000, 5000, 10000]

grid = GridSearchCV(pipe, cv=5, param_grid={
    'col_selection__col_list': [ranking[:i] for i in col_ns]})

grid.fit(X, y_binary_int)

fncs.write_col_ranking(grid.cv_results_['mean_test_score'], col_ns,
                       write_to_desk=True)
