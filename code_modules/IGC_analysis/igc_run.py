import importlib

from code_modules.IGC_analysis import functions as fncs

importlib.reload(fncs)
# Load data
path, file_name = "data/IGC/", "IGC_pfam_bacteriocins.csv"
df_bacteriocins = fncs.process_igc(in_path=f'{path}{file_name}',
                                   separator=';', load_preexisting=True)

# Get long sequences with relatively shorter hits
long_domains = fncs.get_long_domains(in_df=df_bacteriocins,
                                     included_cols=['target name'])
long_domains.to_csv(f'{path}long_seq_hits.csv', sep='\t')
print(long_domains)
