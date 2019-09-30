def make_fasta_str(in_dataframe, description_columns, sequence_column):
    """
    Takes a pandas data frame with columns containing descriptions for
    fasta header and a column containing sequence. Outputs a string in fasta
    format
    """
    return "".join([f">{' '.join([str(e[c]) for c in description_columns])}"
                    f"\n{e[sequence_column]}\n"
                    for _, e in in_dataframe.iterrows()])


if __name__ == '__main__':
    # Convert DNA-recombination_DNA-integration and glycolytic-process_lipid-A-
    # biosynthetic-process to fasta
    import pandas as pd
    import re

    file_paths = [
        "data/uniprot/" + p
        for p in ["glycolytic-process_lipid-A-biosynthetic-process.csv",
                  "DNA-recombination_DNA-integration.csv"]]

    df_list = [pd.read_csv(file_path, sep=',') for file_path in
               file_paths]

    for file_path, df in zip(file_paths, df_list):
        with open(f"{re.sub(file_path.split('.')[-1], 'fasta', file_path)}",
                  'w') as f:
            f.write(make_fasta_str(in_dataframe=df,
                                   description_columns=df.columns[
                                       ~df.columns.isin(["Sequence",
                                                         "old_index"])],
                                   sequence_column='Sequence'))
