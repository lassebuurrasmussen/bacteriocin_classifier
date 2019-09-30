import importlib

import matplotlib

import code_modules.nn_training.functions as fncs

importlib.reload(fncs)
matplotlib.use('module://backend_interagg')  # Allennlp changes backend

# Do one of 'gridsearch', 'cross_validation', 'train_final_model',
# 'test_final_model'
ACTION = 'test_final_model'

if ACTION == 'gridsearch':
    param_dict = dict(conv_filters1=[50, 75],
                      kernel_sizes=[[2, 3, 4], [3, 4, 5], [6, 8, 10]],
                      maps_per_kernel=[2, 3],
                      dropout1=[0],
                      dropout2=[0, 0.2],
                      pool_size=[3],
                      conv_filters2=[300],
                      dense_units=[120],
                      lr=[0.001],
                      epochs=[100],
                      batch_size=[1024])

    results = fncs.hyperparameter_gridsearch(
        i_to_skip=tuple(),
        nn_architechture='CNNPAR', param_dict=param_dict, n_permutations=24,
        kwarg_key_list=['conv_filters1', 'kernel_sizes', 'maps_per_kernel',
                        'dropout1', 'dropout2', 'pool_size', 'conv_filters2',
                        'dense_units', 'lr'])
elif ACTION == 'train_final_model':
    fncs.train_final_model()

elif ACTION == 'cross_validation':
    fncs.run_cross_validation(nn_architechture='CNNPAR',
                              in_kwargs={'conv_filters1': 75,
                                         'conv_filters2': 300,
                                         'dense_units': 120,
                                         'dropout1': 0.0,
                                         'dropout2': 0.0,
                                         'kernel_sizes': [6, 8, 10],
                                         'lr': 0.001,
                                         'maps_per_kernel': 2,
                                         'pool_size': 3})


elif ACTION == 'test_final_model':
    importlib.reload(fncs)
    weights_path = ("code_modules/nn_training/BAC_UNI_len2006/final_elmo_CNNPAR"
                    "_BAC_UNI_len2006/final_elmo_CNNPAR_BAC_UNI_len2006"
                    "_1565079880")
    fncs.train_final_model(just_test_model=True, weights_path=weights_path)

else:
    raise AssertionError
