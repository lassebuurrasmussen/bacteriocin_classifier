from math import floor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import code_modules.encoding.encoding_testing_functions as enct
import code_modules.miscellaneous.csv2fasta as csv2fasta

matplotlib.use('module://backend_interagg')  # Allennlp changes backend
#################################Generate fasta#################################
# Load data
y, y_test, data, data_seq, x_raw = enct.load_data(
    dataset_to_use='hamid', use_mmseqs_cluster=False, max_length=9999999,
    min_length=0)

df = data.loc[y_test.index, ['Sequence', 'Entry name', 'Entry', 'type']]
df['description'] = df['Entry name'] + df['Entry']

# Save fasta
with open("code_modules/ampep/ampep-matlab-code/bacteriocin.fasta",
          'w') as f:
    f.write(csv2fasta.make_fasta_str(df.drop('type', axis=1), ['description'],
                                     'Sequence'))

##############################Load MATLAB results###############################
MATLAB_COMMAND_LINES = """
test_fasta_path = 'bacteriocin.fasta'
[predict_result] = main_function(test_fasta_path)
writetable(predict_result,'bacteriocin_result','WriteRowNames',true)
"""

with open(
        "code_modules/ampep/ampep-matlab-code/bacteriocin_result.txt",
        'r') as f:
    result = f.read().splitlines()

# Convert to pandas data frame
result = (pd.DataFrame([l.split(',') for l in result[1:]], index=df.index,
                       columns=['row_name', 'is_bac', 'score']))
# Convert to float
result[['is_bac', 'score']] = result[['is_bac', 'score']].astype(float)

# Join tables
df = df.merge(result, left_on='description', right_on='row_name', how='inner')

# Assert flawless join
assert not df.isna().any(axis=None)

###############################Calculate accuracy###############################
mean_test_acc = round(((df['is_bac'] == df['type'].map({'UNI': 0, 'BAC': 1})).mean() * 100), 2)
print(mean_test_acc)
# Out[]: 65.42


fig: plt.Figure = plt.figure()

# Accuracy plot
ax1: plt.Axes = plt.axes()
certainties = np.linspace(0, .5, 51)
accuracies = []
distance_from_05 = (.5 - df['score']).abs()
for certainty in certainties:
    subset = df[distance_from_05 >= certainty]
    accuracies.append((subset['is_bac'] == subset['type'].map({'UNI': 0, 'BAC': 1})).mean())
certainties *= 2

ax1.plot(certainties, accuracies, 'b', label='AmPEP accuracy')

ax1.set_xticks(np.arange(0, 1.1, 0.1))
ax1.set_yticks(np.arange(floor(min(accuracies) * 100) / 100, 1.01, 0.02))
# ax1.set_xlim(0, 1)
ax1.grid()
ax1.set_title(f"Test set accuracy - AmPEP: {mean_test_acc / 100:.3f}")
ax1.set_ylabel("Accuracy")
ax1.set_xlabel("Certainty threshold")

# Density plot
twinax: plt.Axes = ax1.twinx()
sns.kdeplot(distance_from_05 * 2, ax=twinax, label='AmPEP certainty density',
            color='red')
twinax.legend().remove()
twinax.set_ylim(top=15)

twinax.set_ylabel("Density")
twinax.set_yticks([])

fig.legend(loc=[0.2, 0.8])
fig.tight_layout()
plt.savefig("code_modules/ampep/test_accuracy")
