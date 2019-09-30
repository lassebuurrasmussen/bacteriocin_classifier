import importlib

import matplotlib
import matplotlib.pyplot as plt

import code_modules.nn_training.functions as fncs
import code_modules.nn_training.grid_explorer_functions as grdf

matplotlib.use('module://backend_interagg')  # Allennlp changes backend
importlib.reload(fncs)

#%% RUN 1
result_df_run1 = grdf.extract_grid_results()

grdf.plot_correlations(result_df=result_df_run1)

grdf.boxplot(x='lr', y='val_acc', hue='conv_filters', col='batch_size',
             row='pool_size', data=result_df_run1)

result_df_run1.val_acc.plot.density()
plt.show()

#%% RUN 2
f_path = ("code_modules/nn_training/BAC_UNI_len2006/gridsearch_log"
          "_RNNCNN1aug.txt")
result_df_run2 = grdf.extract_grid_results(f_path)

grdf.plot_correlations(result_df=result_df_run2)

grdf.boxplot('batch_size',
             'val_acc',
             data=result_df_run2,
             hue='epochs',
             col='lr',
             row='conv_filters')

result_df_run2.val_acc.plot.density()
plt.show()

#%% RUN 3
f_path = ("code_modules/nn_training/BAC_UNI_len2006/gridsearch_log_"
          "RNNCNN1aug-2.txt")
result_df_run3 = grdf.extract_grid_results(f_path)

grdf.boxplot(x='epochs', y='val_acc', hue='lr', col=None, row=None,
             data=result_df_run3, kind='point')

#%%
result_df_run1.val_acc.max()
result_df_run2.val_acc.max()
result_df_run3.val_acc.max()
result_df_run1.iloc[result_df_run1.val_acc.idxmax()].to_dict()
#%%
