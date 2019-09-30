#%%
import re
from io import StringIO
from subprocess import check_output

import pandas as pd


def process_igc(in_path, number_of_header_lines=0, out_path=None,
                separator=';', get_bacteriocins=False, save_file=False,
                load_preexisting=False):
    header = ["target name", "target accession", "tlen", "query name",
              "accession", "qlen", "E-value", "domain_score", "domain_bias",
              "domain number", "total n domains", "c-Evalue", "i-Evalue",
              "score", "bias", "from hmm coord", "to hmm coord",
              "from ali coord", "to ali coord", "from env coord",
              "to env coord", "acc", "description of target"]

    if load_preexisting:
        return pd.read_csv(in_path, sep=separator, header=None, names=header)

    if get_bacteriocins:
        command = f"grep [Bb]acterioc {in_path}".split()
        out_data = check_output(command).decode().splitlines()
        number_of_header_lines = 0

    else:
        # Load data
        with open(in_path, 'r') as f:
            out_data = f.read().splitlines()

    # Format data fields
    out_data = [re.sub(r"\s+", separator, line, 22)
                for line in out_data[number_of_header_lines:]]
    out_data = "\n".join(out_data)

    if save_file:
        with open(out_path, 'w') as f:
            f.write(out_data)

    else:
        return pd.read_csv(StringIO(out_data), sep=';', header=None)


def get_long_domains(in_df, diff_threshold=10, included_cols=None):
    included_cols = (['query name', 'qlen'] if included_cols is None else
                     included_cols + ['query name', 'qlen'])

    # Isolate relevant columns
    long_seqs = in_df[included_cols + [
        'from hmm coord', 'to hmm coord', 'tlen']].copy()

    long_seqs['alignment_length'] = abs(long_seqs['from hmm coord'] -
                                        long_seqs['to hmm coord'])

    long_seqs = long_seqs[(long_seqs['alignment_length'] <
                           long_seqs['tlen'] - diff_threshold)]

    long_seqs['length_diff'] = (long_seqs['tlen'] -
                                long_seqs['alignment_length'])
    long_seqs['length_ratio'] = (long_seqs['alignment_length'] /
                                 long_seqs['tlen']).round(2)

    long_seqs = (long_seqs.sort_values('length_diff').
                 drop(columns=['from hmm coord', 'to hmm coord']).
                 reset_index(drop=True).
                 rename(columns={'qlen': "query_length",
                                 'tlen': "target_length"}))

    return long_seqs


if __name__ == '__main__':
    do_processing = False
    if do_processing:
        process_igc(in_path="data/IGC/IGC_pfam.domtblout", number_of_header_lines=2,
                    out_path="data/IGC/IGC_pfam_bacteriocins.csv",
                    get_bacteriocins=True, save_file=True)
