import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from code_modules.nn_training import tflogs2pandas as tf2pd

tf2pd.dump_to_csv(
    log_dir='code_modules/nn_training/BIDGRU_GLY_LIP_len3096/logs',
    output_dir='code_modules/nn_training/BIDGRU_GLY_LIP_len3096/csv_logs')

dir_path = "code_modules/nn_training/BIDGRU_GLY_LIP_len3096/"

tf2pd.extract_results_from_csv(
    pd_df_path=dir_path, out_file_name='BIDGRU_GLY_LIP_len3096_result.csv')

#%%
# Load training results
df = (pd.read_csv('code_modules/nn_training/BIDGRU_GLY_LIP_len3096/'
                  'BIDGRU_GLY_LIP_len3096_result.csv')
      .query("run_type == 'validation'"))

sns.set()
sns.reset_defaults()

plt.close('all')
fig, ax = plt.subplots(figsize=[5, 3.5])
ax: plt.Axes
# plt.plot('embedding', 'mean', 'b.', markersize=25,
#          data=df)
plt.errorbar('embedding', 'mean', 'std',
             linewidth=4, data=df, capsize=10, capthick=4)
# plt.xticks(rotation=10)
ax.grid()
ax.set_yticks(np.arange(round(ax.get_ylim()[0], 2),
                        round(ax.get_ylim()[1], 2) + 0.01, 0.01))
ax.set_xlabel('Encoding')
ax.set_ylabel('Accuracy')

[ax.text(x + 0.01, y_, f"{s:.3f}")
 for x, y_, s in
 zip(range(len(df['embedding'])), df['mean'],
     df['mean'])]

SAVE_FIG = True

if SAVE_FIG:
    fig.tight_layout()
    fig.savefig(f"paper/figures/BIDGRU_GLY_LIP_result")

else:
    plt.show()
