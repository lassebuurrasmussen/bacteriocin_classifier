import code_modules.nn_training.functions as fncs

fncs.run_cross_validation(nn_architechture='BIDGRU', do_w2v=True,
                          dataset_to_use='GLYLIP')
