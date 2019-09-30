import pandas as pd
from Bio import SeqIO
import numpy as np

path = "data/shortCompleteUniquegenes/"
sequence_path = "shortCompleteUnique_SeqProt_Prodigal_v1.faa"
id_path = "shortCompleteUnique_SeqProt_Prodigal_v1_ID.txt"

fasta_dir = {'ID': [], 'Sequence': [], 'Name': []}
for record in SeqIO.parse(f"{path}{sequence_path}", 'fasta'):
    fasta_dir['ID'].append(record.id)
    fasta_dir['Sequence'].append(str(record.seq))
    fasta_dir['Name'].append(record.description)

data = pd.DataFrame(fasta_dir)
