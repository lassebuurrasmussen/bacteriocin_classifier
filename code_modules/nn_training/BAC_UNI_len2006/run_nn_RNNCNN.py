import code_modules.nn_training.functions as fncs

import importlib

importlib.reload(fncs)

DO_GRIDSEARCH = False
if DO_GRIDSEARCH:
    param_dict = {'lr': [0.00001, 0.0001],
                  'conv_filters': [640],
                  'kernel_size': [26],
                  'pool_size': [26],
                  'lstm_units': [320],
                  'dropout1': [0.2],
                  'dropout2': [0.0],
                  'epochs': [150],
                  'batch_size': [256]}

    results = fncs.hyperparameter_gridsearch(
        nn_architechture='RNNCNN', param_dict=param_dict, n_permutations=6,
        kwarg_key_list=['lr',
                        'conv_filters',
                        'kernel_size',
                        'pool_size',
                        'lstm_units',
                        'dropout1',
                        'dropout2'])
else:
    fncs.run_cross_validation(nn_architechture='RNNCNN', fold_i_to_skip=tuple(),
                              **{'batch_size': 1024,
                                 'epochs': 75,
                                 'in_kwargs': {'conv_filters': 640,
                                               'dropout1': 0.2,
                                               'dropout2': 0.0,
                                               'kernel_size': 26,
                                               'lr': 0.0001,
                                               'pool_size': 26}})
