import pandas as pd
from code_modules.data_preprocessing.preprocessing_functions \
    import test_sequence_column

path = "data/bagel/"
bagel_date = '021419'

data, cols, headers = [], [], []
for CLASS in range(1, 4):
    # Load text file
    data_i = pd.read_csv(path + f'bagel{bagel_date}c{CLASS}.txt', sep='\t')

    data_i.rename(columns={'NCBI.1': 'NCBI_ID',
                           'Uniprot.1': 'Uniprot_ID'}, inplace=True)

    data.append(data_i)

data: pd.DataFrame = pd.concat(data, sort=False)  # Join data frames
data.dropna(how='all', axis=1, inplace=True)  # Drop all-nan columns

test_sequence_column(in_data=data)

save_file = False
if save_file:
    data.to_csv(path + f'bagel{bagel_date}.csv', sep='\t', index=False)

# %%

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    lens = data.Sequence.astype(str).apply(len)
    # lens = lens[lens < lens.sort_values().iloc[-2]]

    fig: plt.Figure = plt.figure()
    fig.set_size_inches(3, 3)
    [plt.hist(lens[data.ID.astype(str).str[-1] == str(c)],
              bins=np.arange(0, 400, 10), alpha=0.5, label=f"{c}")
     for c in [1, 2, 3]]

    plt.legend()

    plt.ylabel("Count")
    plt.xlabel("Sequence Length")
    plt.title("BAGEL Database")

    plt.show()
