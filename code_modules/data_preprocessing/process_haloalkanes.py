# %%
import pandas as pd
import numpy as np
from Bio import SeqIO

# Load fasta records
records = list(SeqIO.parse("data/haloalkanes/all_haloalkanes.fa", "fasta"))

# Extract IDs and sequences
Sequences = pd.Series([str(record.seq) for record in records])
IDs = pd.Series([str(record.id) for record in records])

# Convert to data frame
df = pd.DataFrame({"ID": IDs, "Sequence": Sequences, "Name": IDs})

# Get type
df['type'] = df['ID'].str[0].map({"s": "ground", "c": "ocean"})

# Remove stars and validate removal
df['Sequence'] = df.Sequence.str.strip("*")
assert (df.Sequence.str.extract("([*])").fillna(39210) == 39210).all(axis=None)

# Save to csv
df.to_csv("data/haloalkanes/all_haloalkanes.csv", index=False)

# %%
if __name__ == '__main__':
    # Visualize length distributions
    import matplotlib.pyplot as plt

    lens = df.Sequence.apply(len)

    bins = np.linspace(0, lens.max(), 30)
    [plt.hist(lens.loc[df[df.type == t].index],
              color={"ground": "red", "ocean": 'blue'}[t], alpha=0.4, bins=bins,
              label=t) for t in df.type.unique()]

    plt.legend()
    plt.show()
