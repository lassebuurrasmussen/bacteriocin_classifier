import pickle

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os

import code_modules.encoding.aa_encoding_functions as enc


def make_len_cat_col(in_data_short, verbose, show_plots):
    # Make length categorical variable
    n_cats = len(in_data_short) // 50
    in_data_short['length_cat'] = pd.qcut(in_data_short.length,
                                          n_cats,
                                          duplicates='drop',
                                          labels=False)

    # Ensure that all stratification groups are larger than 1
    while (in_data_short.groupby(["type", "length_cat"]).size() < 2).any():
        n_cats //= 2
        in_data_short['length_cat'] = pd.qcut(in_data_short.length,
                                              n_cats,
                                              duplicates='drop',
                                              labels=False)

    if verbose:
        if show_plots:
            in_data_short['length_cat'].value_counts(sort=False).plot(kind='bar')
            plt.title("Length categories")
            plt.show()

    return in_data_short


def data_split(in_data, verbose=False, max_length=100, min_length=10,
               show_plots=False, force_new_split=False, allow_splitsave=None):
    # Define split file name based on labels and row count
    split_file_str = (f"data/trte_splits/"
                      f"{'_'.join(np.sort(in_data.type.unique()))}_"
                      f"{len(in_data)}.txt")

    # If there exists a split file load it unless not allowed
    if os.path.isfile(split_file_str) and not force_new_split:
        print("Loading predefined split")
        with open(split_file_str, 'rb') as f:
            lab_split = pickle.load(f)
        return lab_split

    if verbose:
        print("Making new split")
        if show_plots:
            # Show lengths
            in_data.hist(
                column='length', by='type', layout=[-1, 1], figsize=[7, 5],
                bins=np.arange(0, 1e2, 5), density=False)
            plt.gcf().suptitle("Length distribution",
                               horizontalalignment='right')
            plt.show()
        print(f"Length describtions:")
        print(in_data.groupby('type').length.describe())
        print(f"Sequences between min and max:")
        print(in_data.groupby('type').length.apply(
            lambda xi: ((xi > min_length) & (xi < max_length)).sum()))

    # Assert that no indices are missing
    assert (np.arange(len(in_data)) == in_data.index).all()

    # Exclude sequences outside length limit
    in_data_short = in_data[
        (in_data.length < max_length) &
        (in_data.length > min_length)].copy()

    if verbose:
        if show_plots:
            in_data_short.hist(column='length', by='type', layout=[2, 1],
                               bins=np.arange(min_length, 1e2, 1))
            plt.gcf().suptitle("Length distribution subset",
                               horizontalalignment='right')
            plt.show()

    # Make length categorical column for stratification
    in_data_short = make_len_cat_col(in_data_short, verbose, show_plots)

    # Get indices for training set
    lab_split = train_test_split(in_data_short.type,
                                 test_size=0.2, random_state=63,
                                 stratify=in_data_short[[
                                     'type', 'length_cat']])

    if allow_splitsave:
        with open(split_file_str, 'wb') as f:
            pickle.dump(lab_split, f)

    elif allow_splitsave is None:
        raise AssertionError("Please allow or disallow split saving")

    return lab_split


def load_data(dataset_to_use, use_mmseqs_cluster=False, verbose=True,
              show_plots=False, force_new_split=False, max_length=100,
              allow_splitsave=None, min_length=10):
    datasets = {
        'GLYLIP': "data/uniprot/glycolytic-process_lipid-A-biosyntheti"
                  "c-process.csv",
        'DNA_Rec_Int': "data/uniprot/DNA-recombination_DNA-integration"
                       ".csv",
        'halo': "data/haloalkanes/all_haloalkanes.csv",
        'ANIPLA': "data/camp/CAMP_Animalia_Viridiplantae.csv",
        'hamid': 'data/hamid_reproduction/hamid_data_set.csv'
    }

    csv_path = (datasets[dataset_to_use] if dataset_to_use in datasets.keys()
                else dataset_to_use)

    in_data = pd.read_csv(csv_path)

    if use_mmseqs_cluster:
        if verbose:
            print("Applying MMSeqs clusters")

        try:
            clusters = pd.read_csv(f"code_modules/mmseqs_clustering/DB_"
                                   f"{dataset_to_use}_clu.tsv", sep='\t',
                                   index_col=0, names=['Entry', 'identifier'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Please run MMseqs2 for {dataset_to_use}")

        in_data = in_data.set_index('Entry').reindex(clusters.index.unique()). \
            reset_index()

    # Get lengths
    in_data['length'] = in_data.Sequence.apply(len)

    # Split into test and training stratified by length and type ratio
    y, y_test = data_split(in_data=in_data,
                           verbose=verbose,
                           show_plots=show_plots,
                           force_new_split=force_new_split,
                           max_length=max_length,
                           allow_splitsave=allow_splitsave,
                           min_length=min_length)

    in_data = in_data[in_data.index.isin(np.concatenate([y.index, y_test.index]))]

    data_seq = enc.expand_seqstr_series(in_data.Sequence)

    x = data_seq.loc[y.index]

    return y, y_test, in_data, data_seq, x
