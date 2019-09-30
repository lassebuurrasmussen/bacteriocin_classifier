import re
import time
from itertools import chain

import numpy as np
import pandas as pd
from Bio.Alphabet.IUPAC import IUPACProtein
from gensim import matutils
from gensim.models import Word2Vec

from code_modules.encoding.word2vec_generator import Word2VecGenerator

letters = list(IUPACProtein().letters)  # Get AA letters


def get_slice(in_iter, lower, upper, in_len):
    lower_upper = [lower, upper]

    lower_upper = [lim if lim > 0 else 0 for lim in lower_upper]
    lower_upper = [lim if lim <= in_len else in_len for lim in lower_upper]

    return in_iter[lower_upper[0]: lower_upper[1]]


def isolate_inpt_oupt(in_sequences, in_x_sequence_indices, k=3, w=5):
    half_slice_size = (k - 1) + (w * k)
    seq_slices = []
    for s, x_idx in zip(in_sequences, in_x_sequence_indices):
        sequence_x_tuples = []
        for xi in x_idx:
            s_len = len(s)

            limits = [
                (xi - half_slice_size, xi),  # Left input kmers
                (xi - (k - 1), xi + (k - 1) + 1),  # Output kmers containing X
                (xi + 1, xi + half_slice_size)]  # Right input kmers

            sequence_x_tuples.append(tuple(
                get_slice(s, lim[0], lim[1], s_len)
                for lim in limits))
        seq_slices.append(sequence_x_tuples)

    return seq_slices


def isolate_x_sequences(in_sequences):
    # Isolate sequences containing X
    x_sequences_index, x_sequences = zip(
        *[(i, s) for i, s in enumerate(in_sequences) if 'X' in s])

    # Pad sequences for regex expression
    x_sequences_padded = [f'Å{s}Å' for s in x_sequences]

    # Remove sequences only containing Xs that are duplicated
    max_x_repeats = 2
    pattern = re.compile(f'(?=[^X](X{{1,{max_x_repeats}}})[^X])')
    sequences_x_matches = [list(re.finditer(pattern, s)) for i, s in
                           zip(x_sequences_index, x_sequences_padded)]

    # Extract the indices of each matching X per sequence
    sequences_x_indices = [list(chain(*[{m.start(1) - 1, m.end(1) - 2}
                                        for m in r if m.start(1)]))
                           for r in sequences_x_matches]

    # Filter away sequences with duplicated Xs
    x_sequences_index, x_sequences, sequences_x_indices = zip(
        *[(x_sequences_index[i], x_sequences[i], x_indices)
          for i, x_indices in enumerate(sequences_x_indices)
          if x_indices])

    return x_sequences_index, x_sequences, sequences_x_indices


def replace_x_with_qmark(in_words, w, in_sequences, in_sequence_x_indices):
    seqs_x_target_qm = []
    for i, entry in enumerate(in_words):
        entry_list = []

        for wi, word in enumerate(entry):
            if len(word) == w:
                new_word = f'{word[:w // 2]}?{word[-w // 2 + 1:]}'

            else:
                entry_len = len(in_sequences[i])
                x_idx = in_sequence_x_indices[i][wi]

                if x_idx == 0:
                    new_word = f'?{word[1:]}'

                elif entry_len / 2 > x_idx:
                    new_word = f'{word[:x_idx]}?{word[x_idx + 1:]}'

                else:
                    x_idx_right = entry_len - x_idx
                    new_word = f'{word[:x_idx_right]}?{word[x_idx_right + 1:]}'

            entry_list.append(new_word)

        seqs_x_target_qm.append(entry_list)

    return seqs_x_target_qm


def get_context_and_target_dfs(in_sequences, sequence_x_indices, k, w):
    # Isolate sequence fragments sorrounding X as context and containing X as
    # target
    cntxt_targt_tuples = isolate_inpt_oupt(in_sequences, sequence_x_indices)

    # Convert the sequences of only the context tuples to all their possible
    # kmers and save the reading frame
    cntxt_kmers = [[list(chain(*[[(i % 3, word[i:i + k])
                                  for i in range(len(word) - (k - 1))]
                                 for word_i, word in enumerate(tup)
                                 if word_i != 1]))
                    for tup in seq_tuples] for seq_tuples in cntxt_targt_tuples]

    # Convert to Pandas Data Frame
    columns = ['seq_idx', 'x_n', 'kmer_n', 'reading_frame', 'kmer']
    rows = []
    for e_i, entry in enumerate(cntxt_kmers):
        for x_i, x_kmers in enumerate(entry):
            for k_i, rf_kmer in enumerate(x_kmers):
                rows.append([e_i, x_i, k_i, rf_kmer[0], rf_kmer[1]])

    # Have kmers as list in data frame
    df_cntxt = (pd.DataFrame(rows, columns=columns).
                groupby(['seq_idx', 'x_n', 'reading_frame']).
                apply(lambda grp: grp['kmer'].tolist()).rename('context').
                reset_index())

    # Select target sequences
    seqs_x_target = [[tup[1] for tup in seq_tuples]
                     for seq_tuples in cntxt_targt_tuples]

    # For each entry, replace the X in question with a questionmark
    seqs_x_target_qmark = replace_x_with_qmark(
        in_words=seqs_x_target, w=w, in_sequences=in_sequences,
        in_sequence_x_indices=sequence_x_indices)

    # For each X target sequence, find all possible sequences with ? substituted
    seqs_x_target_mut = [[[word.replace('?', l) for l in letters]
                          for word in seq_entry]
                         for seq_entry in seqs_x_target_qmark]

    # Split each mutation to kmers
    target_kmers = [[[[word[i:i + k] for i in range(len(word) - (k - 1))]
                      for word in variations] for variations in entry]
                    for entry in seqs_x_target_mut]

    # Convert to Pandas Data Frame
    columns = ['seq_idx', 'x_n', 'x_idx', 'mutation_n',
               'kmer_n', 'full_fragment', 'kmer']
    rows = []
    for e_i, entry in enumerate(target_kmers):
        for v_i, variations in enumerate(entry):
            for w_i, words in enumerate(variations):
                for k_i, kmer in enumerate(words):
                    rows.append([e_i, v_i, sequence_x_indices[e_i][v_i], w_i,
                                 k_i, seqs_x_target_qmark[e_i][v_i], kmer])

    df_target = pd.DataFrame(rows, columns=columns)

    # Calculate reading frame
    df_target['x_idx_fragment'] = df_target['full_fragment'].str.index('?')
    df_target['x_idx_mod3'] = df_target['x_idx'] % 3
    df_target['fragment_reading_frame'] = (df_target['x_idx_mod3'] -
                                           df_target['x_idx_fragment']) % 3
    df_target['reading_frame'] = (df_target['fragment_reading_frame'] +
                                  df_target['kmer_n']) % 3

    df_target = (df_target[['seq_idx', 'x_n', 'mutation_n', 'kmer_n', 'kmer',
                            'reading_frame']].
                 groupby(['seq_idx', 'x_n', 'reading_frame']).
                 apply(lambda grp: grp['kmer'].tolist()).rename('mutations').
                 reset_index())

    df = pd.merge(df_cntxt, df_target, how='right',
                  on=['seq_idx', 'x_n', 'reading_frame'])

    return df


def select(in_df, seq=0, x=0, rf=0):
    return in_df[(in_df['seq_idx'] == seq) & (in_df['x_n'] == x) &
                 (in_df['reading_frame'] == rf)]


def make_grp(in_df, cols=('seq_idx', 'x_n', 'reading_frame')):
    return in_df.groupby(list(cols))


def select_idxs(in_df, cols=('seq_idx', 'x_n', 'reading_frame')):
    return in_df[list(cols)]


def predict_from_context(context, model):
    return (pd.Series(*list(zip(*model.predict_output_word(
        context, topn=11_000)))[::-1], name='propability').reindex(context).
            reset_index())


def predict_output_word(model, context_words_list, topn=10, do_sorting=True,
                        possible_mutations=None, vocab_indices=None):
    """Modified function from method of Word2Vec class from gensim library"""
    word_vocabs = [model.wv.vocab[w] for w in context_words_list
                   if w in model.wv.vocab]

    word2_indices = [word.index for word in word_vocabs]

    l1 = np.sum(model.wv.vectors[word2_indices], axis=0)
    if word2_indices and model.cbow_mean:
        l1 /= len(word2_indices)

    # propagate hidden -> output and take softmax to get probabilities
    prob_values = np.exp(np.dot(l1, model.trainables.syn1neg.T))
    prob_values /= sum(prob_values)

    if do_sorting:
        top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)

    else:
        top_indices = list(range(len(prob_values)))[:topn]

    if not possible_mutations and not vocab_indices:
        # returning the most probable output words with their probabilities
        return [(model.wv.index2word[index1], prob_values[index1])
                for index1 in top_indices]
    elif vocab_indices:
        return [prob_values[index1] for index1 in vocab_indices]
    else:
        return [prob_values[index1] for index1 in top_indices
                if model.wv.index2word[index1] in possible_mutations]


def predict_output_words(model, context_words_list, topn=10, do_sorting=True,
                         possible_mutations=None, vocab_indices=None):
    """Modified function from method of Word2Vec class from gensim library"""
    word_vocabs = [[model.wv.vocab[w] for w in words_list
                    if w in model.wv.vocab] for words_list in context_words_list]

    word2_indices = [[word.index for word in word_vocab]
                     for word_vocab in word_vocabs]

    l1 = [np.sum(model.wv.vectors[word2_indices_i], axis=0)
          for word2_indices_i in word2_indices]

    for i, word2_indices_i in enumerate(word2_indices):
        if word2_indices_i and model.cbow_mean:
            l1[i] /= len(word2_indices_i)

    # propagate hidden -> output and take softmax to get probabilities
    l1 = np.stack(l1)
    prob_values = np.exp(np.dot(l1, model.trainables.syn1neg.T))
    prob_values /= prob_values.sum(1)[:, None]

    if do_sorting:
        top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)

    else:
        top_indices = list(range(prob_values.shape[1]))[:topn]

    if not possible_mutations and not vocab_indices:
        # returning the most probable output words with their probabilities
        return [(model.wv.index2word[index1], prob_values[index1])
                for index1 in top_indices]
    elif vocab_indices:
        return [[prob_values[i, index1] for index1 in vocab_indices_i]
                for i, vocab_indices_i in enumerate(vocab_indices)]
    else:
        return [prob_values[index1] for index1 in top_indices
                if model.wv.index2word[index1] in possible_mutations]


def get_word_probabilities(context, possible_mutations, vocab, word2vec_model):
    vocab_indices = vocab.reindex(possible_mutations)['index'].tolist()

    return predict_output_word(word2vec_model, context, do_sorting=False,
                               topn=11000, vocab_indices=vocab_indices)


def get_probable_mutations(probability_df, in_sequences, sequence_x_indices):
    most_likely_mutations = (probability_df.apply(pd.Series).
                             groupby(['seq_idx', 'x_n']).sum().idxmax(1).
                             reset_index(name='most_likely_mutation'))

    most_likely_mutations = most_likely_mutations.groupby(
        ['seq_idx'])['most_likely_mutation'].apply(list).reset_index()
    most_likely_mutations['sequence'] = np.array(in_sequences)[
        most_likely_mutations['seq_idx']]
    most_likely_mutations['x_idx'] = np.array(sequence_x_indices)[
        most_likely_mutations['seq_idx']]

    return most_likely_mutations


def replace_x(in_row):
    sequence = in_row['sequence']
    x_idxs = in_row['x_idx']
    in_letter_indices = in_row['most_likely_mutation']

    replacers = [letters[i] for i in in_letter_indices]

    for i, x_idx in enumerate(x_idxs):
        sequence = f'{sequence[:x_idx]}{replacers[i]}{sequence[x_idx + 1:]}'

    return sequence


def get_vocab_indices(df, word2vec_model):
    # Extract vocabulary and add index column
    vocab = Word2VecGenerator.extract_vocab_vectors(word2vec_model)
    vocab['index_in_vocab'] = range(len(vocab))
    vocab = vocab[['index_in_vocab']]

    # List all required kmers and the index of the sequence it belongs to
    indicer = pd.Series(*list(zip(
        *[(mut, i) for i, seq_muts in enumerate(df['mutations'])
          for mut in seq_muts])), name='mutation').reset_index()

    # Extract required kmers and add index which they belong to
    vocab_indices = vocab.reindex(indicer['mutation']).reset_index()
    vocab_indices['index'] = indicer['index']

    # Group by sequence belonging
    return (vocab_indices.groupby('index')['index_in_vocab'].apply(list).
            tolist())


def impute(in_sequences, k=3, w=5):
    start_time = time.time()
    print('Isolating sequences containing X'.center(80, '_'))
    print('Elapsed time:', f'{round(time.time() - start_time, 2)} seconds')
    # Isolate sequences containing X and the index of those who do
    index_containing_x, seqs_x, seqs_x_idxs = isolate_x_sequences(in_sequences)

    print('Getting X coordinates and neighborhood'.center(80, '_'))
    print('Elapsed time:', f'{round(time.time() - start_time, 2)} seconds')
    # Convert to neural network input
    df = get_context_and_target_dfs(
        in_sequences=seqs_x, sequence_x_indices=seqs_x_idxs, k=k, w=w)

    print('Extracting Word2Vec vectors'.center(80, '_'))
    print('Elapsed time:', f'{round(time.time() - start_time, 2)} seconds')
    # Load word2vec model
    word2vec_model = Word2Vec.load("data/hamid_wordvec_model_trembl_size_200")
    vocab_indices = get_vocab_indices(df=df, word2vec_model=word2vec_model)

    print('Calculating substitution probabilities'.center(80, '_'))
    print('Elapsed time:', f'{round(time.time() - start_time, 2)} seconds')
    df['probability'] = predict_output_words(
        model=word2vec_model, context_words_list=df['context'].tolist(),
        topn=11_000, do_sorting=False, vocab_indices=vocab_indices)

    assert (make_grp(df).size() == 1).all()
    probability_df = make_grp(df)['probability'].first()

    # Get most likely mutation agreeing with all reading frames
    most_likely_mutations = get_probable_mutations(probability_df, seqs_x,
                                                   seqs_x_idxs)

    print('Replacing Xs'.center(80, '_'))
    print('Elapsed time:', f'{round(time.time() - start_time, 2)} seconds')
    # Replace X in all sequences
    seqs_x_removed = most_likely_mutations.apply(replace_x, axis=1).tolist()

    for x_seq_index, index in enumerate(index_containing_x):
        in_sequences[index] = seqs_x_removed[x_seq_index]

    return in_sequences


if __name__ == '__main__':
    from code_modules.word2vec_clustering.functions import \
        load_uniprot

    # Load sequences
    data = load_uniprot(n_each=None)
    seqs = data['Sequence'].tolist()

    seqs_imputed2 = impute(seqs)
