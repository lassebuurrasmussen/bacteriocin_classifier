import code_modules.data_preprocessing.uniprot_dl_functions as dl_funcs
import importlib

importlib.reload(dl_funcs)
# %% Bacteria GO terms for encoding test
go_searches = [["nitrogen utilization", 19740],
               ["polysaccharide metabolic process", 5976],
               ["carbohydrate metabolic process", 5975]]

for go_search in go_searches[-1:]:
    if len(go_search) != 0:
        dl_funcs.download_uniprot(go_word=go_search[0], go_id=go_search[1],
                                  do_call=True)

dl_funcs.download_uniprot("primary metabolic process", 44238)

# %% Different taxa with same GO term
taxa = ['Bacteria',
        'Viridiplantae',
        'Homo sapiens',
        'Pan',
        'Cercopithecidae',
        'Mus musculus',
        'Hymenoptera',
        'Viruses']

for taxon in taxa:
    dl_funcs.download_uniprot(go_word="carbohydrate metabolic process",
                              go_id=5975, taxonomy=taxon, len_max=255,
                              limit=1_000, taxon_in_fname=True, add_taxon_col=True)
