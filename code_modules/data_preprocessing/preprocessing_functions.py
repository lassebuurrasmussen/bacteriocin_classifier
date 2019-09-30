import re

import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Alphabet.IUPAC import IUPACProtein


def sspace(in_string, rep='%20'): return in_string.replace(' ', rep)


def test_sequence_column(in_data, na_threshold=3, seq_col_name='Sequence',
                         id_col_name='ID'):
    # Split sequence
    split_seqs = in_data[seq_col_name].str.split("", expand=True)

    # Obtain present characters
    characters_present = np.unique(split_seqs.values.
                                   flatten().astype(str))

    # Load real AAs and append empty and X
    aas = pd.Series(list(IUPACProtein.letters) + ['', 'X'])

    # Identify the present characters that are not AAs
    non_aas = characters_present[~np.isin(characters_present, aas)]

    # Make sure that no sequences contain these
    assert len(in_data[(split_seqs.isin(non_aas)).any(1)][id_col_name]) == 0, \
        'illegal AA characters present'

    # Identify empty sequence rows
    na_rows = in_data[in_data[seq_col_name].isna()][id_col_name]
    if len(na_rows) > na_threshold:
        raise AssertionError(f"Number of NA rows above threshold of "
                             f"{na_threshold}.\nNA rows:\n{na_rows}")


def get_newest_filename(in_names):
    pattern = re.compile(r"([0-9]{6})\.csv")
    dates = [re.findall(pattern, name)[0] for name in in_names]

    idx_newest = pd.to_datetime(pd.Series(dates)).values.argmax()

    return in_names[idx_newest]


def is_within_range(in_length, lower=10, upper=359):
    return (in_length >= lower) & (in_length <= upper)


def process_refseq_assembly_file(index, input_file):
    # Give each line a 5 digit index
    index = str(index).zfill(5)

    # Extract sequences and descriptions from each record
    sequence_list = []
    description_list = []
    for rec in SeqIO.parse(input_file, 'fasta'):
        if is_within_range(len(rec)):
            sequence_list.append(f'{index}{str(rec.seq)}')
            description_list.append(f'{index}{rec.description}')

    return sequence_list, description_list


def process_seq_batch(batch_i, minibatch_size):
    print(f"Batch {batch_i + 1} of 5")

    path = "data/refseq/processed_refseq/"
    batch_patch = f"all_sequences_processing_step2_batch{batch_i}.dump"

    sequences = joblib.load(path + batch_patch)

    minibatch_start_idxs = list(range(0, len(sequences), minibatch_size))
    path += "minibatches/"
    for i, minibatch_start_i in enumerate(minibatch_start_idxs):
        if not i % 1000:
            print(f"Minibatch {i + 1} of {len(minibatch_start_idxs)}")

        joblib.dump(
            sequences[minibatch_start_i:minibatch_start_i + minibatch_size],
            f"{path}batch{batch_i}minibatch_{i}")


def fasta2csv(path):
    fasta_dir = {'ID': [], 'Sequence': [], 'Name': []}
    for record in SeqIO.parse(path, 'fasta'):
        fasta_dir['ID'].append(record.id)
        fasta_dir['Sequence'].append(str(record.seq))
        fasta_dir['Name'].append(record.description)
    return pd.DataFrame(fasta_dir)
