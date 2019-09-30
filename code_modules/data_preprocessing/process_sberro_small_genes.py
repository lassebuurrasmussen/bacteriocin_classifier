import pandas as pd

import code_modules.data_preprocessing.preprocessing_functions as pfncs
import importlib

importlib.reload(pfncs)
#################################load all genes#################################
data_path = 'data/sberro_small_genes/'
data = pfncs.fasta2csv(f"{data_path}Data File 1.faa")

if all([ID in Name for ID, Name in zip(data.ID, data.Name)]):
    print("Dropping redundant ID column")
    data.drop('ID', axis=1, inplace=True)

data.to_csv(f"{data_path}sberro_small_genes_all.csv", index=False)

##########################load representative clusters##########################
cols_to_use = ['Small protein family ID',
               'Identified in bacteria',
               'Identified in in eukaryota',
               'Identified in archaea',
               'Identified in viruses',
               'Cluster representative',
               'Is family predicted to be antimicrobial',
               'Confidence score in antimicrobial peptide',
               'AA sequence of clusters representative']
df = pd.read_excel(f"{data_path}1-s2.0-S0092867419307810-mmc5.xlsx",
                   sheet_name="4,539 small protein families",
                   usecols=cols_to_use)

df.rename(columns={c: c.replace(' ', '_') for c in cols_to_use}, inplace=True)
df.rename(columns={'AA_sequence_of_clusters_representative': 'Sequence'},
          inplace=True)

pfncs.test_sequence_column(
    df, seq_col_name='Sequence',
    id_col_name='Small_protein_family_ID')

df.to_csv(f'{data_path}sberro_small_genes.csv')
