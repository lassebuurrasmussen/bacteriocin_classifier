import importlib

import code_modules.nn_training.functions as fncs

importlib.reload(fncs)

DO_GRIDSEARCH = False
if DO_GRIDSEARCH:
    param_dict = {'l1_units': [32, 64, 320],
                  'l2_units': [32, 64, 320],
                  'dropout1': [0.0, 0.2, 0.5],
                  'dropout2': [0.0, 0.2, 0.5],
                  'regularizer_vals': [0, 0.01, 0.001],
                  'epochs': [20, 50, 100],
                  'batch_size': [32, 32 * 4, 32 * 8]}

    results = fncs.hyperparameter_gridsearch(param_dict=param_dict)

else:
    fncs.run_cross_validation(nn_architechture='DNN',
                              in_kwargs=dict(dropout1=0.5,
                                             dropout2=0.5,
                                             l1_units=32,
                                             l2_units=64,
                                             regularizer_vals=0),
                              epochs=50,
                              batch_size=256)
