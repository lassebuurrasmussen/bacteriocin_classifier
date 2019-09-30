# %%
import pandas as pd


def argmax(l): return max(range(len(l)), key=lambda i: l[i])


def _go_term_processing(data, do_save, verbose, choosen_gos,
                        custom_abbreviations, output_title):
    # Split up GO terms
    go_terms = data['Gene ontology (biological process)'].str.split(";", expand=True)

    # Name each column
    go_terms.rename(columns={c: f"go_term{c}" for c in go_terms.columns},
                    inplace=True)

    # Combine go terms to one columns
    go_terms_combined = pd.concat([go_terms[c] for c in go_terms.columns]). \
        rename('go_terms').dropna().apply(lambda term: term.strip())

    # Join go terms on index with data without go terms
    data: pd.DataFrame = pd.merge(data.drop(columns=['Gene ontology (biological process)']),
                                  go_terms_combined, how='right', left_index=True,
                                  right_index=True)

    data = data.reset_index().rename(columns={'index': "old_index"})

    counts = data["go_terms"].value_counts()[::-1]
    if verbose:
        # Show counts
        print(counts)

    if not choosen_gos:
        # Isolate high occurrence
        data = data[data["go_terms"].isin(counts[-3:-1].index)].copy()
    else:
        assert isinstance(choosen_gos, list)
        data = data[data["go_terms"].isin(
            [counts[go:go + 1].index[0] for go in choosen_gos])].copy()

    # Drop duplicates
    data: pd.Series = data.loc[data["Entry"].drop_duplicates(keep=False).index]

    if verbose:
        # Recount after drop
        print(data["go_terms"].value_counts())

    # Abbreviate go terms and rename column
    if not custom_abbreviations:
        map_series = pd.Series([s[:3].upper() for s in
                                data["go_terms"].unique()],
                               index=data["go_terms"].unique())

        if map_series.duplicated().any():
            raise AssertionError("Please enter unique abbreviations")

    else:
        map_series = pd.Series(custom_abbreviations,
                               index=data.go_terms.unique())

    data["go_terms"] = data["go_terms"].map(map_series)
    data.rename(columns={"go_terms": "type"}, inplace=True)

    if verbose:
        print(map_series.values)
        print(f"Out data shape: {data.shape}")

    out_file_name = "_".join(map_series.index.str.extract("(.+)\[",
                                                          expand=False).str.
                             strip().str.replace(" ", "-").str.replace(',', ''))
    out_file_name = f"data/uniprot/{out_file_name}{output_title}.csv"

    if verbose:
        print(f"Out file name: {out_file_name}")
    if do_save:
        data.to_csv(out_file_name, index=False)


def _taxa_processing(data, do_save, in_file_path, output_title):
    data = data.drop(columns='Gene ontology (biological process)').rename(
        columns={'Taxonomic lineage (ALL)': 'type'})

    taxa_all = data['type'].str.split(',', expand=True)

    # Find taxonomic level all have in common
    taxa_intersection = list(set(taxa_all.values[0]).intersection(
        *[set(r) for r in taxa_all.values]))

    if len(taxa_intersection) > 1:
        # Find the lowest shared taxonomic level
        taxa_hierachy = taxa_all.iloc[0].tolist()
        order = [taxa_hierachy.index(taxon) for taxon in taxa_intersection]

        taxa = taxa_intersection[argmax(order)]
    else:
        taxa = taxa_intersection[0]

    tr_dict = {ord(' '): '_', ord('('): '', ord(')'): ''}
    taxa = taxa.strip().translate(tr_dict)

    data['type'] = taxa

    out_file_name = f'{in_file_path}{taxa}{output_title}.csv'

    if do_save:
        data.to_csv(out_file_name, index=False)

    return data


def prepare_raw_uniprot(in_file_path, in_file_name, do_save, verbose=False,
                        choosen_gos=None, custom_abbreviations=None,
                        ignore_gos=False, output_title='', return_df=False):
    # Read in sequences
    data = pd.read_csv(in_file_path + in_file_name, "\t")

    if ignore_gos:
        data = _taxa_processing(data=data, do_save=do_save,
                                in_file_path=in_file_path,
                                output_title=output_title)

        if return_df:
            return data

    else:
        _go_term_processing(
            data=data, do_save=do_save, verbose=verbose,
            choosen_gos=choosen_gos, custom_abbreviations=custom_abbreviations,
            output_title=output_title)


# %% Bacteria GO terms for encoding test

file_path = "data/uniprot/"

prepare_raw_uniprot(
    in_file_path=file_path,
    in_file_name="uniprot_goa_carbohydrate_metabolic_process_280219-1249.txt",
    do_save=False, verbose=True)

prepare_raw_uniprot(
    in_file_path=file_path,
    in_file_name="uniprot_goa_primary_metabolic_process_280219-1414.txt",
    do_save=False, verbose=True, choosen_gos=[-4, -5],
    custom_abbreviations=['DNI', 'DNR'])

# %% Different taxa with same GO term
file_path = "data/uniprot/"
files = ['Bacteria_270519-1245', 'Cercopithecidae_270519-1246',
         'Homo_sapiens_270519-1245', 'Hymenoptera_270519-1246',
         'Mus_musculus_270519-1246', 'Pan_270519-1246',
         'Viridiplantae_270519-1245', 'Viruses_270519-1246']
files = [f'uniprot_goa_carbohydrate_metabolic_process_{f}.txt' for f in files]

dfs = []
for file in files:
    dfs.append(prepare_raw_uniprot(in_file_path=file_path, in_file_name=file,
                                   do_save=False, ignore_gos=True,
                                   return_df=True))

pd.concat(dfs).to_csv(f'{file_path}carbohydrate_metabolic_process_many_taxa.csv',
                      index=False)
