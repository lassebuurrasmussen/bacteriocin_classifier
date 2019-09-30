from gensim.models import Word2Vec
import pandas as pd


class Word2VecGenerator:
    def __init__(self, corpus_path=None):
        if corpus_path is not None:
            self.corpus = self.load_corpus(corpus_path)
            self.corpus_kmers = self.corpus2kmers(self.corpus)
            self.model = self.train_model(self.corpus_kmers)
            self.vectors = self.extract_vocab_vectors(self.model)

    @staticmethod
    def load_file(in_path):
        return pd.read_csv(in_path, sep='\t', usecols=['Sequence'])

    def load_corpus(self, in_corpus_path):
        # Convert to list of path strings
        if type(in_corpus_path) != list:
            assert type(in_corpus_path) == str
            in_corpus_path = [in_corpus_path]

        # Load CSV files, concatenate them and drop NAN values
        out_corpus = pd.concat([self.load_file(in_path) for in_path in in_corpus_path],
                               ignore_index=True).dropna().drop_duplicates()

        # Convert to sequence list
        return out_corpus['Sequence'].tolist()

    @staticmethod
    def corpus2kmers(in_corpus, k=3):
        # For each sequence get kmer spanning from every third index i to i + k,
        # starting from 0, 1, ..., k-1. Stop at index n - (k-1) to cut away kmers of
        # length < k. Output format:
        #                        [[seq1 rf1], [seq1 rf2] ... [seqN rf2], [seqN rf3]]
        return [[seq[i:i + k] for i in range(start, len(seq) - (k - 1), 3)]
                for seq in in_corpus for start in [0, 1, 2]]

    @staticmethod
    def train_model(in_sentences):
        return Word2Vec(sentences=in_sentences, size=200, window=5, min_count=0,
                        sg=1, workers=3)

    @staticmethod
    def extract_vocab_vectors(model):
        word_indices = {v.index: k for k, v in model.wv.vocab.items()}
        df_vectors = pd.DataFrame(model.wv.vectors)
        df_vectors.index = df_vectors.index.map(word_indices)

        return df_vectors


# %%

if __name__ == '__main__':
    w2v = Word2VecGenerator(corpus_path=['data/bagel/bagel021419.csv',
                                         'data/bactibase/bactibase021419.csv'])
    vectors = w2v.vectors
