import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code_modules.word2vec_clustering.functions import get_bacteriocins


def load_bacteriocins(lower_length_cutoff=30):
    out_df = get_bacteriocins(
        n_each=None, camp_source=None, drop_non_bacteriocins=True,
        drop_bagel_bactibase=False, type_by_camp_belonging=False,
        upper_length_cutoff=489)
    out_df['length'] = out_df['Sequence'].apply(len).astype(int)
    out_df = out_df[out_df['length'] >= lower_length_cutoff]

    return out_df


def load_negative_set(in_data_path, in_filename):
    out_df = pd.read_csv(in_data_path + in_filename, '\t')

    out_df['length'] = out_df['Sequence'].apply(len).astype(int)

    return out_df


def take_subset_of_negative_set(in_negative_set, in_bacteriocins):
    lengths = pd.merge(in_negative_set['length'].value_counts().sort_index(),
                       in_bacteriocins['length'].value_counts().sort_index(),
                       how='right', left_index=True, right_index=True, suffixes=['_neg', '_bac'])

    assert not (pd.Series(lengths['length_bac'] > lengths['length_neg'])).any()

    np.random.seed(70)
    neg_entries_filtered = pd.Series()
    for length in lengths.index:
        n_seqs = lengths.loc[length, 'length_bac']

        entries_i = in_negative_set.loc[in_negative_set['length'] == length,
                                        'Entry'].sample(n_seqs)

        neg_entries_filtered = neg_entries_filtered.append(entries_i)

    out_negative_set_filtered = in_negative_set[
        in_negative_set['Entry'].isin(neg_entries_filtered)].copy()

    return out_negative_set_filtered


def concat_data_sets(in_bacteriocins, in_negative_set):
    in_bacteriocins['type'] = 'BAC'
    in_bacteriocins['Entry name'] = in_bacteriocins['ID']
    in_bacteriocins = (in_bacteriocins.rename(columns={'ID': 'Entry'}).
                       drop(columns=['bacteriocin', 'class']))

    in_negative_set['type'] = 'UNI'
    in_negative_set.drop(columns=['Gene ontology (biological process)',
                                  'Organism'], inplace=True)

    out_df = (pd.concat([in_negative_set, in_bacteriocins], sort=False).
              rename_axis(index='old_index').reset_index())

    return out_df


if __name__ == '__main__':
    data_path = 'data/hamid_reproduction/'
    filename = 'hamid_negative_set_unfiltered.tab'

    negative_set = load_negative_set(in_data_path='data/hamid_reproduction/',
                                     in_filename='hamid_negative_set_unfiltered.tab')
    bacteriocins = load_bacteriocins(lower_length_cutoff=10)
    negative_set_filtered = take_subset_of_negative_set(
        in_negative_set=negative_set, in_bacteriocins=bacteriocins)

    full_set = concat_data_sets(in_bacteriocins=bacteriocins,
                                in_negative_set=negative_set_filtered)

    DO_SAVE = True
    if DO_SAVE:
        # negative_set_filtered.to_csv(data_path + 'hamid_negative_set.csv',
        #                              index=False)
        # bacteriocins.to_csv(data_path + 'hamid_positive_set.csv', index=False)

        full_set.to_csv(data_path + "hamid_data_set.csv", index=False)

    DO_PLOT = False
    if DO_PLOT:
        matplotlib.use('module://backend_interagg')  # Allennlp changes backend

        f, axs = plt.subplots(3, 1, sharey='all', sharex='all', figsize=[12, 8])

        negative_set['length'].rename('Negative set').plot(
            kind='density', ax=axs[0])

        negative_set_filtered['length'].rename('Negative set filtered').plot(
            kind='density', ax=axs[1])

        bacteriocins['length'].rename('Bacteriocin set').plot(
            kind='density', ax=axs[2])

        [ax.legend() for ax in axs]

        plt.show()
