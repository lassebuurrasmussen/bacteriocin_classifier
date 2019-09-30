# %%
import importlib
import os
from time import time as ti

import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

import code_modules.encoding.aa_encoding_functions as enc
import code_modules.encoding.encoding_testing_functions as enct

os.chdir("/home/wogie/Documents/KU/Bioinformatics_Master/Block_3/Master Thesis")
# %% Load data sets
importlib.reload(enct)
importlib.reload(enc)

datasets = ['GLYLIP', 'DNA_Rec_Int', 'halo', 'ANIPLA']
dataset_i = 0
use_mmseqs_cluster = True
y, y_test, data, data_seq, X = enct.load_data(
    dataset_to_use=datasets[dataset_i], use_mmseqs_cluster=use_mmseqs_cluster)
use_mmseqs_cluster = 'mmseqs' if use_mmseqs_cluster else ""
# %% Cross validation setup
importlib.reload(enct)
importlib.reload(enc)

Max_length = data.length.max()

Model_name = f'{"_".join(sorted(data.type.unique()))}_{len(data)}{use_mmseqs_cluster}'

make_encoding_models = False
if make_encoding_models:
    enc.atchley_encode(in_df=data_seq, save_model=True, model_name=Model_name + "_atchley",
                       get_fractions=False)
    enc.w2v_embedding_cluster_encode(data_seq, save_model=True,
                                     model_name=Model_name + "_WE",
                                     get_fractions=False)
    enc.fastdna_encode(in_df=data_seq, in_data=data, model_name=Model_name,
                       use_premade=False, just_train_model=True)

encoded_xs = enc.get_all_enc(model_name=Model_name, in_x=X, max_len=Max_length,
                             dropfastdna=True)

# Define grid of parameters
param_grid = [{'alpha': (1.0000000000000001e-05, 9.9999999999999995e-07)}]

# %% Run CV
random_state = 43
start_time = ti()
grids = {}
for encoding_type, x_encoded in encoded_xs.items():
    grid = GridSearchCV(SGDClassifier(tol=1e-3, random_state=random_state,
                                      penalty='l2', n_jobs=4),
                        param_grid=param_grid, cv=10, verbose=1,
                        return_train_score=True)
    grid.verbose = (3 if encoding_type in ['w2v_embedding',
                                           'atchley',
                                           'fastdna',
                                           'elmo_embedding']
                    else grid.verbose)
    print(f"Fitting {encoding_type}..")
    grid.fit(x_encoded, y)
    grids[encoding_type] = grid
    print(f"done, score={grid.best_score_}")

end_time = ti()
print(end_time - start_time)

# %% Save model
save_model = True
if save_model:
    print("Saving grid searched model...")
    joblib.dump(
        grids, f"code_modules/saved_models/gridsearchmodel070319{Model_name}")
