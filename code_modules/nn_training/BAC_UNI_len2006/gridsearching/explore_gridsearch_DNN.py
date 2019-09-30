import importlib

import matplotlib
import pandas as pd

import code_modules.nn_training.functions as fncs
import code_modules.nn_training.grid_explorer_functions as grdf

matplotlib.use('module://backend_interagg')  # Allennlp changes backend
importlib.reload(fncs)

result_df = pd.read_csv("code_modules/nn_training/temp_log30jul.csv")

#%%


grdf.plot_correlations(result_df=result_df)

#%%
grdf.boxplot(x='regularizer_vals', y='val_acc', hue='batch_size', col=None,
             row=None, data=result_df)
