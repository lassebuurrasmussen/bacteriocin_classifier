import os
import subprocess
from itertools import product
from pathlib import Path
from time import time as ti
from typing import Union

import matplotlib
import numpy as np
import pandas as pd
from Bio.Alphabet import Reduced
from Bio.Alphabet.IUPAC import IUPACProtein
from allennlp.commands.elmo import ElmoEmbedder
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder

from code_modules.miscellaneous.csv2fasta import make_fasta_str
from code_modules.encoding.word2vec_generator import Word2VecGenerator


# os.chdir("/home/wogie/Documents/KU/Bioinformatics_Master/Block_3/Master Thesis")
# matplotlib.use('module://backend_interagg')  # Allennlp changes backend


def reduced_alphabet_encode(in_df, alphabet='m10'):
    alphs = {'hp': Reduced.hp_model_tab,
             'm10': Reduced.murphy_10_tab,
             'm15': Reduced.murphy_15_tab,
             'm4': Reduced.murphy_4_tab,
             'm8': Reduced.murphy_8_tab}

    # Remove X values
    out_df = find_and_replace_xs(in_df)

    # Transform into reduced alphabet
    out_df = out_df.replace(alphs[alphabet])

    # Extract the used alphabet for one-hot encoding
    predefined_alphabet = pd.Series(alphs[alphabet]).unique()

    # Add nan value to alphabet
    predefined_alphabet = np.append(predefined_alphabet, 'nan').reshape(1, -1)

    # One-hot encode
    return onehot_encode(out_df, predefined_alphabet=predefined_alphabet)


def extract_atchley_factors(in_series):
    # Load atchley factors
    atchley_factors = pd.read_csv('data/atchley.txt', sep='\t'). \
        set_index("amino.acid")

    atchley_factors = atchley_factors.append(
        pd.DataFrame(np.zeros([1, 5]), index=['X'],
                     columns=atchley_factors.columns))

    return pd.Series(atchley_factors.loc[in_series.dropna()].values.
                     flatten(order='C'))


def make_atchley_kmer_table(get_locally=True, k=3):
    if get_locally is False:
        # Make new CSV file
        letters = list(IUPACProtein().letters)  # Get AA letters
        kmers = list(product(letters, repeat=k))  # Make all kmers

        # Make k-mers data frame
        kmers_df = pd.DataFrame(kmers)
        kmers_df.index = ["".join(k) for k in kmers]
        kmers_df = kmers_df.apply(extract_atchley_factors, axis=1)
        try:
            kmers_df.to_csv("data/atchley_kmers.csv", sep='\t')
        except FileNotFoundError:
            raise AssertionError(f"atchley_kmers.csv not found. Run function"
                                 f" 'make_atchley_kmer_table' with parameter "
                                 f"get_locally=False")

    else:
        # Load pre-made CSV file
        kmers_df = pd.read_csv("data/atchley_kmers.csv", sep='\t', index_col=0)

    return kmers_df


def replace_string_xs(in_series, window_size=7):
    # Find all Xs in string
    x_indxs = np.where(in_series == 'X')[0]

    # Get the window around the Xs
    window_idxs = [[max(x - window_size, 0),
                    min(x + window_size + 1, len(in_series))]
                   for x in x_indxs]

    windows = [in_series[a[0]:a[1]] for a in window_idxs]

    # Remove Xs from the windows
    windows = [w[w != 'X'] for w in windows]

    # Get most frequent AA
    replacers = [w.value_counts().idxmax() for w in windows]

    # Replace Xs with most frequent AA in window
    for x_indx, r in zip(x_indxs, replacers):
        in_series.iloc[x_indx] = r

    return in_series


def find_and_replace_xs(in_df):
    # Get indices of rows containing X
    idxs = pd.unique(np.where(in_df == 'X')[0])

    if len(idxs) != 0:
        # Replace the Xs with most frequent AA in window
        out_df = in_df.copy()
        out_df.iloc[idxs] = in_df.iloc[idxs].apply(replace_string_xs, axis=1)

        return out_df
    else:
        return in_df


def get_cluster_memberships(in_data, n_clusters=100, get_locally=True,
                            save_model=False, model_name="",
                            random_seed=43):
    # Cluster the AAs based on their Atchley factors
    model_path = (f"code_modules/saved_models/kmeans_model_{model_name}"
                  f"{n_clusters}clusters.pkl")

    if get_locally:

        try:
            kmeans = joblib.load(model_path)
        except FileNotFoundError:
            raise AssertionError(f"{model_path} not found. Run function "
                                 f"'make_kmer_atchley_clusters' with parameter "
                                 f"get_locally=False")

        assert kmeans.n_clusters == n_clusters

    else:
        kmeans = KMeans(n_clusters, random_state=random_seed)
        kmeans.fit(in_data)

        if save_model:
            joblib.dump(kmeans, model_path)

    # Return group membership for each AA
    return pd.Series(kmeans.predict(in_data), index=in_data.index)


def str2kmers(in_string, k=3):
    """Takes an array with a string and a padding value and returns a list of
    k-mers of size parameter k with trailing nan values specified by pad
    value"""

    out_kmers = [in_string[0][i:i + k] for i in range(len(in_string[0]) - (k - 1))]

    out_kmers = np.pad(out_kmers, pad_width=(0, int(in_string[1])),
                       mode='constant', constant_values=np.nan)

    return out_kmers


def seq_pddf_2_npstrs(in_df):
    out_arr = in_df.values.copy()

    # Convert the array to a long str (seqs separated by ",")
    full_str = "".join([entry for subset in out_arr.tolist()
                        for entry in subset + [',']])

    # Drop the trailing separator if it exists
    full_str = full_str[:-1] if full_str[-1] == ',' else full_str

    # Remove nan values
    out_arr_seqstr = np.array(full_str.replace('nan', '').split(','))

    return out_arr_seqstr.reshape(-1, 1)


def get_sample_kmers(in_df, k=3, get_unique: Union[bool, str] = True,
                     input_is_str=False):
    # Get sequence as strings
    if input_is_str:
        in_data_seqstr = in_df
    else:
        in_data_seqstr = seq_pddf_2_npstrs(in_df)

    # Get sequence lengths for aligning kmer array shapes
    lenghts = pd.Series(in_data_seqstr.flatten()).apply(len) - (k - 1)
    max_len = lenghts.max()
    padding_n = max_len - lenghts

    all_kmers = np.array([str2kmers([seq, pad], k=k) for seq, pad in
                          zip(in_data_seqstr.flatten(), padding_n)])

    if get_unique is True:
        all_kmers = all_kmers.flatten()
        return pd.unique(all_kmers[all_kmers != 'nan'])

    elif get_unique == 'both':
        all_kmers_unique = all_kmers.flatten()
        return all_kmers, pd.unique(all_kmers_unique[all_kmers_unique != 'nan'])

    else:
        return all_kmers


def get_group_fractions(sample_kmers, kmer_groupings, n_clusters=100):
    """Takes an array of kmers and computes fraction belonging to each
    cluster"""
    # Map each kmer to its cluster
    sample_clustered = pd.DataFrame(sample_kmers).apply(
        lambda row: row.map(kmer_groupings), 1).fillna(n_clusters).astype(int)

    # Append NaN cluster to each sample (else bincount wont broadcast)
    sample_clustered_fractions = np.append(
        sample_clustered.values, np.repeat(
            n_clusters, len(sample_clustered)).reshape(-1, 1), 1)

    # Count members of each cluster per sample
    sample_clustered_fractions = np.apply_along_axis(
        np.bincount, 1, sample_clustered_fractions)[:, :-1]

    # Get the fraction of the total k-mers in each sample
    sample_clustered_fractions = (
            sample_clustered_fractions /
            sample_clustered_fractions.sum(1).reshape(-1, 1))

    return sample_clustered_fractions


def w2v_embedding_encode(in_df, word2vec_model=None, k=3,
                         input_max_length=None, input_is_flat_series=False,
                         original_shape=None):
    if input_is_flat_series:
        in_kmers = in_df
    else:
        in_kmers = get_sample_kmers(in_df=in_df, get_unique=False)

    if word2vec_model is None:
        # Load Hamid Word2Vec model. (Contains 'X', 'U', 'B', 'Z', 'O' AA words)
        word2vec_model = Word2Vec.load("data/hamid_wordvec_model_trembl_size_200")

    # Extract vocabulary vectors
    vocab_vectors = Word2VecGenerator.extract_vocab_vectors(word2vec_model)

    # Add index for nan
    vocab_vectors = vocab_vectors.append(pd.DataFrame(np.zeros([1, 200]),
                                                      index=['nan']))
    # Map each kmer to an index
    index = pd.Series(range(len(vocab_vectors)),
                      index=vocab_vectors.index.copy())

    # Translate all kmers into their indices for advanced numpy indexing
    if input_is_flat_series:
        out_shape = original_shape
        flat_kmers = in_kmers.copy()
    else:
        out_shape = in_kmers.shape
        flat_kmers = pd.Series(in_kmers.flatten().copy())

    uniq_kmers = pd.Series(flat_kmers.unique())

    kmers_not_in_vocab = uniq_kmers[(~uniq_kmers.isin(index.index.unique()))]

    if len(kmers_not_in_vocab) > 0:
        to_be_replaced = flat_kmers.isin(kmers_not_in_vocab)
        print(f"{len(kmers_not_in_vocab)} kmers do not exist in Word2Vec "
              f"vocabulary. These are found "
              f"{to_be_replaced.sum()} places in the "
              f"sequences being encoded and will be replaced with nan")

        flat_kmers[to_be_replaced] = 'nan'

    in_kmers_indx = (flat_kmers.map(index).values.
                     reshape(out_shape))

    # Get context vector for each row
    vvv = vocab_vectors.values  # Use numpy array for advanced indexing
    out_array = vvv[in_kmers_indx].reshape(out_shape[0], out_shape[1] * 200)

    if input_max_length:
        # Pad array to have appropriate size
        max_length = (input_max_length - (k - 1)) * 200

        assert max_length >= out_array.shape[1]
        padding = np.zeros([len(out_array), max_length - out_array.shape[1]])
        out_array = np.c_[out_array, padding]

    return out_array


def expand_seqstr_series(in_series):
    # Save original index
    out_indx = in_series.index.copy()

    # Get all string lengths
    str_lengths = in_series.apply(len)

    # Get padding needed for all string to have same lengths
    out_padding = str_lengths.max() - str_lengths

    # Add padding as a column
    out_df = pd.DataFrame(in_series)
    out_df['padding'] = out_padding

    # Expand string to arrays for single letters
    return pd.DataFrame(
        np.apply_along_axis(
            lambda row: np.array(list(row[0]) + [np.nan] * row[1]),
            axis=1, arr=out_df), index=out_indx)


def onehot_encode(in_df, use_predefined_alph=True, predefined_alphabet=None,
                  return_encoder=False):
    if use_predefined_alph:
        if predefined_alphabet is None:
            all_letters = np.array(
                [['D', 'X', 'I', 'G', 'V', 'R', 'E', 'W', 'Q', 'T', 'Y', 'K',
                  'S', 'H', 'P', 'F', 'A', 'nan', 'L', 'C', 'N', 'M']],
                dtype='<U3')
        else:
            all_letters = predefined_alphabet
    else:
        # Get unique letters
        all_letters = np.array(list({entry for sublist in
                                     in_df.apply(list, axis=1).tolist()
                                     for entry in sublist}), ndmin=2)

    # Define categories in each position in sequence as all letters
    oh_categories = np.repeat(all_letters, in_df.shape[1], axis=0).tolist()

    # One hot encode
    oh = OneHotEncoder(categories=oh_categories)
    out_df = oh.fit_transform(in_df.values)

    if return_encoder:
        return out_df, oh
    else:
        return out_df


def map_kmers_to_values(max_length, sample_kmers, kmer_map, k,
                        factors_per_kmer):
    # Get vector lenghts and required paddings
    vec_lens = (np.array([len(_l[_l != 'nan']) for _l in sample_kmers]) *
                factors_per_kmer)

    if max_length is None:
        paddings = max(vec_lens) - vec_lens

    else:
        paddings = (max_length - (k - 1)) * factors_per_kmer - vec_lens

    # Flatten kmers and extract all atchley vectors
    sample_kmersflat = sample_kmers.flatten()
    sample_kmersflat = sample_kmersflat[sample_kmersflat != 'nan']

    temp = kmer_map.reindex(sample_kmersflat).values

    # Get kmer counts per sample
    kmer_count_cumsum = np.append(0, vec_lens // factors_per_kmer).cumsum()

    # reshape and pad to obtain sample athcley encoded kmers
    encoded_array = np.array(
        [np.pad(temp[kmer_count_cumsum[li]:kmer_count_cumsum[li + 1]].
                flatten(), (0, _p), 'constant', constant_values=np.nan)
         for li, _p in zip(range(len(kmer_count_cumsum)), paddings)])

    return np.nan_to_num(encoded_array)  # Replace nan with zero


def atchley_encode(in_df, n_clusters=100, get_locally=False, save_model=False,
                   model_name='', random_seed=43, get_fractions=True,
                   cluster=True, input_max_length=None, k=3):
    # With Xs replaced
    in_df_no_x: pd.Series = find_and_replace_xs(in_df)

    # Get all AA k-mers and their atchley factors
    atchley_kmers = make_atchley_kmer_table()

    sample_kmers, sample_kmers_unique = get_sample_kmers(in_df=in_df_no_x,
                                                         get_unique='both')
    sample_atchley_kmers = atchley_kmers.reindex(sample_kmers_unique)

    if not cluster:
        return map_kmers_to_values(
            max_length=input_max_length, sample_kmers=sample_kmers,
            kmer_map=sample_atchley_kmers, k=k, factors_per_kmer=15)

    kmer_groupings = get_cluster_memberships(
        in_data=sample_atchley_kmers, n_clusters=n_clusters, get_locally=get_locally,
        save_model=save_model, model_name=model_name, random_seed=random_seed)

    if get_fractions:
        sample_clustered_fractions = get_group_fractions(
            sample_kmers=sample_kmers, kmer_groupings=kmer_groupings,
            n_clusters=n_clusters)

        return sample_clustered_fractions


def w2v_embedding_cluster_encode(in_df, word2vec_model=None, k=3, n_clusters=100,
                                 get_locally=False, save_model=False,
                                 model_name='', random_seed=43,
                                 get_fractions=True):
    if word2vec_model is None:
        # Load Hamid Word2Vec model. (Contains 'X', 'U', 'B', 'Z', 'O' AA words)
        word2vec_model = Word2Vec.load("data/hamid_wordvec_model_trembl_size_200")

    # Get entire vocabulary from Word2Vector model
    vocab = np.array(list(word2vec_model.wv.vocab))

    # Get word vector for each vocabulary entry
    vocab_vectors = pd.DataFrame(
        np.apply_along_axis(lambda word: word2vec_model.wv.get_vector(word[0]),
                            axis=1, arr=vocab.reshape(-1, 1)), index=vocab)

    # Select only word vectors for words present in sample
    sample_kmers, sample_kmers_unique = get_sample_kmers(in_df, k,
                                                         get_unique='both')
    sample_vocab_vectors = vocab_vectors.loc[sample_kmers_unique]

    # Cluster
    kmer_groupings_we = get_cluster_memberships(
        in_data=sample_vocab_vectors, n_clusters=n_clusters,
        get_locally=get_locally, save_model=save_model, model_name=model_name,
        random_seed=random_seed)

    if get_fractions:
        out_array = get_group_fractions(sample_kmers=sample_kmers,
                                        kmer_groupings=kmer_groupings_we,
                                        n_clusters=n_clusters)

        return out_array


def get_all_enc(model_name, in_x, max_len, dropfastdna=False,
                encoding_keep_list=None, cuda_device=-1):
    encoder_df = pd.DataFrame(pd.Series(
        {'atchley_cluster': atchley_encode,
         'atchley': atchley_encode,
         'onehot': onehot_encode,
         'reduced_alphabet': reduced_alphabet_encode,
         'w2v_embedding_cluster': w2v_embedding_cluster_encode,
         'w2v_embedding': w2v_embedding_encode,
         'fastdna': fastdna_encode,
         'elmo_embedding_summed': elmo_embedding_encode,
         'elmo_embedding': elmo_embedding_encode
         }, name='transformer'))

    # Warning - this data frame is very large and cannot be printed#############
    encoder_df = pd.concat(
        [encoder_df, pd.Series(
            {'atchley_cluster': {'in_df': in_x, 'get_locally': True,
                                 'model_name': model_name + "_atchley"},

             'atchley': {'in_df': in_x, 'cluster': False,
                         'input_max_length': max_len},

             'onehot': {'in_df': in_x},

             'reduced_alphabet': {'in_df': in_x},

             'w2v_embedding_cluster': {'in_df': in_x, 'get_locally': True,
                                       'model_name': model_name + "_WE"},

             'w2v_embedding': {'in_df': in_x, 'input_max_length': max_len},

             'fastdna': {'in_df': in_x, 'model_name': model_name,
                         'input_max_length': max_len},

             'elmo_embedding_summed': {'in_df': in_x,
                                       'input_max_length': max_len,
                                       'do_sum': True,
                                       'cuda_device': cuda_device},

             'elmo_embedding': {'in_df': in_x, 'input_max_length': max_len,
                                'cuda_device': cuda_device}
             }, name='kwargs')], axis=1)

    if dropfastdna:
        encoder_df.drop(index='fastdna', inplace=True)

    if encoding_keep_list is not None:
        encoder_df = encoder_df[encoder_df.index.isin(encoding_keep_list)]

    encoded_xs = {}
    for i, (transformer, kwargs) in enumerate(zip(encoder_df['transformer'],
                                                  encoder_df['kwargs'])):
        print(f"Encoding {encoder_df.index[i]} ({i + 1} of {len(encoder_df)})"
              f"... ", end="", flush=True)
        start = ti()
        encoded_xs[encoder_df.index[i]] = transformer(**kwargs)
        end = ti()
        print(f"Done ({end - start:.1f} seconds)")

    return encoded_xs


def aas2nucleotides(in_df):
    # Load codon mapper
    codon_table: pd.Series = pd.read_csv(
        "data/AA_to_codon_table.txt", skiprows=1, header=None, index_col=0,
        names=['letter', 'codon']).iloc[:, 0]
    codon_table = codon_table.append(pd.Series({'nan': 'nan'}))

    # Apply map to obtain codons from AAs
    data_seq_nucl = in_df.apply(lambda row: row.map(codon_table))

    # Compress dataframe to series of strings
    nucleotide_strs = np.array(
        ["".join(d.astype(str).tolist()).replace('nan', '') for d in
         data_seq_nucl.values])

    return nucleotide_strs


def fastdna_encode(in_df, model_name, use_premade=True, in_data=None,
                   just_train_model=False, input_max_length=None):
    in_data_nucleotides = aas2nucleotides(in_df=in_df)

    embedding_model_path = f"code_modules/saved_models/fastdna{model_name}.vec"
    k, vector_dimension = 10, 10

    # If FastDNA vocabulary vectors have not already been generated
    model_exists = os.path.isfile(embedding_model_path)

    if model_exists is False or use_premade is False:
        if use_premade:
            raise AssertionError("No premade FastDNA model found, run function "
                                 "with use_premade=False")

        # Make a new data frame with nucleotides instead of AA sequences
        out_data = in_data.copy()
        out_data['Sequence_nuc'] = in_data_nucleotides
        out_data.drop(columns=['Sequence'], inplace=True)

        desc_cols = (['Entry', 'Entry name', 'type']
                     if not model_name.split('_')[:-1] == ['ground', 'ocean']
                     else ['ID', 'Name', 'type'])

        fastdna_train_data = make_fasta_str(
            in_dataframe=out_data,
            description_columns=desc_cols,
            sequence_column='Sequence_nuc')

        fastdna_labels = "\n".join(in_data.type.tolist())

        os.makedirs('temp_f123/output/model')

        # Write out files inputting into FastDNA
        for file, value in zip(['train_data.fasta', 'labels.taxid'],
                               [fastdna_train_data, fastdna_labels]):
            with open(f"temp_f123/{file}", 'w') as f:
                f.write(value)

        # Run FastDNA
        command = (f"code_modules/fastDNA/fastdna supervised "
                   f"-input temp_f123/train_data.fasta "
                   f"-labels temp_f123/labels.taxid "
                   f"-output temp_f123/output/model "
                   f"-minn {k} "
                   f"-dim {vector_dimension} "
                   f"-epoch 1 "
                   f"-thread 4")
        subprocess.check_call(command, shell=True)

        # Extract vocabulary vectors from FastDNA output
        vectors = pd.read_csv('temp_f123/output/model.vec', sep=' ',
                              header=None, skiprows=1,
                              index_col=0)

        # Save vocabulary vectors and remove FastDNA output
        command2 = (f"cp temp_f123/output/model.vec {embedding_model_path}; "
                    f"rm -rf temp_f123")
        subprocess.check_call(command2, shell=True)

    # Else, load premade vocabulary vectors
    else:
        if just_train_model:
            return None
        vectors = pd.read_csv(embedding_model_path, sep=' ', header=None, skiprows=1,
                              index_col=0)

    if just_train_model:
        return None

    # Remove last column if it contains all NAs
    if vectors.iloc[:, -1].isna().all():
        vectors = vectors.iloc[:, :-1]

    sample_kmers = get_sample_kmers(
        in_df=in_data_nucleotides, k=10, get_unique=False,
        input_is_str=True)

    encoded_kmers = map_kmers_to_values(
        max_length=input_max_length * 3 if input_max_length is not None else input_max_length,
        sample_kmers=sample_kmers,
        kmer_map=vectors, k=k, factors_per_kmer=vector_dimension)

    return encoded_kmers


def get_window_kmers(in_df, k=3, w=10, get_windows=True):
    # Get list of strings
    in_sequences = in_df.tolist()

    # Get all kmers for each sequence
    kmers_per_seq = [[x[i:i + k] for i in range(len(x) - (k - 1))]
                     for x in in_sequences]

    if not get_windows:
        return kmers_per_seq

    # Get all windows for each sequence
    wind_per_seq = [[kmers[i:i + w] for i in range(len(kmers) - (w - 1))
                     if len(kmers) >= w] for kmers in kmers_per_seq]

    # Flatten list of windows
    wind_all = [window for windows in wind_per_seq
                for window in windows if window]

    # Get index indicating which sequence each window belongs to
    wind_index = [i for i, windows in enumerate(wind_per_seq)
                  for _ in windows]

    # Convert to a long list of individual kmers for embedding
    kmers_all = pd.Series([kmer for l in wind_all for kmer in l])

    return wind_all, wind_index, kmers_all


def w2v_embedding_window_encode(in_data, in_df, k=3, w=10):
    assert (in_data.index == np.arange(len(in_data))).all()
    wind_all, wind_index, kmers_all = get_window_kmers(in_df=in_df, k=k, w=w)

    out_x = w2v_embedding_encode(in_df=kmers_all, input_is_flat_series=True,
                                 original_shape=[len(wind_all), w])

    # Map indices to types
    out_y = pd.Series(wind_index).map(in_data['type'])

    return out_x, out_y


def elmo_embedding_encode(in_df, input_max_length=None, do_sum=False,
                          cuda_device=-1):
    """
    :param in_df:
    :param input_max_length:
    :param do_sum:
    :param cuda_device: -1 to not use GPU and 0 to use GPU
    :return:
    """
    # Load model
    model_dir = Path('data/elmo_model_uniref50/')
    weights = model_dir / 'weights.hdf5'
    options = model_dir / 'options.json'
    seqvec = ElmoEmbedder(options, weights, cuda_device=cuda_device)

    # Get embeddings
    in_df_str = seq_pddf_2_npstrs(in_df).flatten().tolist()
    embedding = list(seqvec.embed_sentences([list(s) for s in in_df_str]))

    # Pad and flatten embeddings
    flattener = ((lambda e: e.sum(0).flatten()) if do_sum
                 else lambda e: e.flatten())
    embedding_flat = [flattener(e) for e in embedding]

    multiplier = 1024 if do_sum else 3 * 1024

    pad_n = input_max_length * multiplier
    embedding_flat_padded = [np.pad(array=e, pad_width=[0, pad_n - len(e)],
                                    mode='constant', constant_values=0)
                             for e in embedding_flat]

    return np.stack(embedding_flat_padded)
