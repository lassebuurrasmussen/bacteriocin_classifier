import importlib

import code_modules.nn_training.functions as fncs

importlib.reload(fncs)
fncs.run_cross_validation(nn_architechture='CNNPARLSTM')
