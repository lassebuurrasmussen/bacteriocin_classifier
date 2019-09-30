import importlib
import random

import numpy as np

import code_modules.encoding.encoding_testing_functions as enc
import code_modules.word2vec_imputation.impute as imp

importlib.reload(imp)
_, _, data, _, _ = enc.load_data(dataset_to_use='DNA_Rec_Int',
                                 max_length=np.inf,
                                 allow_splitsave=False, min_length=0)

# Drop rows already containing X
data = data[~data['Sequence'].str.contains('X')]
seqs = data['Sequence'].tolist()
lengths = data['length'].tolist()


def mutate_sequences(in_sequences, in_lengths):
    random.seed(648732)
    mutations_per_seq = [random.sample(range(l), k=l // 10) for l in in_lengths]

    out_seqs_mutated = []
    for seqi, mutations in enumerate(mutations_per_seq):
        mutated_seq = in_sequences[seqi]
        for mutation in mutations:
            mutated_seq = (f'{mutated_seq[:mutation]}X'
                           f'{mutated_seq[mutation + 1:]}')

        out_seqs_mutated.append(mutated_seq)

    return out_seqs_mutated


seqs_mutated = mutate_sequences(in_sequences=seqs, in_lengths=lengths)

#%%
importlib.reload(imp)
seqs_imputed = imp.impute(in_sequences=seqs_mutated[:20])
print(seqs_mutated)
