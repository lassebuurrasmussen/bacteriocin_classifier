import code_modules.data_preprocessing.preprocessing_functions as pfnc
import code_modules.encoding.encoding_testing_functions as enct

df2 = pfnc.fasta2csv("code_modules/NeuBI/bacteriocin.fasta")
result = pfnc.fasta2csv("code_modules/NeuBI/bacteriocin.fasta_results")

y, y_test, data, data_seq, x_raw = enct.load_data(
    dataset_to_use='hamid', use_mmseqs_cluster=False, max_length=9999999,
    min_length=0)

df = data.loc[y_test.index, ['Sequence', 'Entry name', 'Entry', 'type']]
df['description'] = df['Entry name'] + df['Entry']

result[['ID', 'bac_score']] = result['ID'].str.split("|", expand=True)

df = df.merge(result, left_on='description', right_on='ID', how='outer')
df['bac_score'] = df['bac_score'].astype(float)

# Three rows that NeuBI doesn't classify because they are longer than 302 aas
df[df.isna().any(1)]

df.dropna(inplace=True)

assert ~df.isna().any(axis=None)

round(((df['bac_score'] > 0.95).astype(int) == df['type'].map(
    {'UNI': 0, 'BAC': 1})).mean() * 100, 2)  # 85.96%
