import importlib

import matplotlib
import matplotlib.pyplot as plt

import code_modules.nn_training.functions as fncs
import code_modules.nn_training.grid_explorer_functions as grdf

matplotlib.use('module://backend_interagg')  # Allennlp changes backend
importlib.reload(fncs)
importlib.reload(grdf)

#%% RUN 1
path = ("code_modules/nn_training/BAC_UNI_len2006/gridsearching/gridsearch_"
        "log_CNNPAR03aug.txt")
result_df_run1 = grdf.extract_grid_results(path_str=path)
result_df_run1 = result_df_run1.drop(columns=['batch_size', 'epochs'])

result_df_run1['kernel_sizes'] = result_df_run1['kernel_sizes'].astype(str).map(
    {'[2, 3, 4]': 1,
     '[3, 4, 5]': 2,
     '[6, 8, 10]': 3})

# Drop the one bad apple because I can't tell why it's failing
# result_df_run1 = result_df_run1[result_df_run1.index != 37]

# grdf.plot_correlations(result_df=result_df_run1)
#
grdf.boxplot(x='pool_size', y='val_acc', hue='conv_filters2', col='dense_units',
             row='lr', data=result_df_run1.query("lr != 0.01"),
             kind='bar',
             min_y=result_df_run1.query("lr != 0.01").val_acc.min() - 0.05,
             max_y=result_df_run1.query("lr != 0.01").val_acc.max() + 0.01,
             )

result_df_run1.val_acc.plot.density()
plt.show()

#%% RUN 2

path = ("code_modules/nn_training/BAC_UNI_len2006/gridsearching/"
        "gridsearch_log_CNNPAR03aug-2.txt")
result_df_run2 = grdf.extract_grid_results(path_str=path)
result_df_run2 = result_df_run2[['conv_filters1', 'dropout2', 'kernel_sizes',
                                 'maps_per_kernel', 'train_acc', 'train_error',
                                 'val_acc', 'val_error']]

result_df_run2['kernel_sizes'] = result_df_run2['kernel_sizes'].astype(str).map(
    {'[2, 3, 4]': 1,
     '[3, 4, 5]': 2,
     '[6, 8, 10]': 3})

# grdf.plot_correlations(result_df_run2)

# grdf.boxplot('maps_per_kernel',
#              'val_acc',
#              'kernel_sizes',
#              'conv_filters1',
#              'dropout2',
#              data=result_df_run2, )

result_df_run2.val_acc.plot.density()
plt.show()
