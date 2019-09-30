import pandas as pd
from code_modules.miscellaneous.csv2fasta import make_fasta_str

path = 'data/camp/'
df = pd.read_csv(path + 'camp_database03-26-19.csv',
                 usecols=['CAMP_ID', 'Taxonomy', 'Sequence'])

df = df[df['Taxonomy'].str.contains("Animalia", na=False) |
        df['Taxonomy'].str.contains("Viridiplantae", na=False)]

# Type hinting for code completion
df: pd.DataFrame = df.copy()
# print("\n".join([f"df[\'{c}\']: pd.Series = df[\'{c}\']" for c in df.columns]))
df['CAMP_ID']: pd.Series = df['CAMP_ID']
df['Taxonomy']: pd.Series = df['Taxonomy']
df['Sequence']: pd.Series = df['Sequence']

# Select first entry in taxonomy column as type
df['type'] = df['Taxonomy'].str.split(',', expand=True).iloc[:, 0]
df['Length'] = df['Sequence'].apply(len)
df = df.drop(columns=['Taxonomy']).rename(columns={'CAMP_ID': "Entry"})

df['Entry name'] = df['Entry']

df['Sequence'] = df['Sequence'].str.replace(" ", "X")
df['Sequence'] = df['Sequence'].str.replace("Z", "X")
df['Sequence'] = df['Sequence'].str.replace("B", "X")

# Drop rows with 5 or more Xs in a row
df = df[~df['Sequence'].str.contains("X{5,}")]

out_file_name = "CAMP_Animalia_Viridiplantae"
df.to_csv(path + out_file_name + '.csv', index=False)

with open(path + out_file_name + '.fasta', 'w') as f:
    f.write(make_fasta_str(df, ['Entry', 'type', 'Length'], 'Sequence'))
