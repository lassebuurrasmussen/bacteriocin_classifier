##########################remove redundant assemblies###########################
import os

import pandas as pd

# Load files
with open("data/refseq/refseq_assembly_hierachy/all_assemblies.txt", 'r') as f:
    directory_structures = f.read().splitlines()
with open("data/refseq/refseq_assembly_hierachy/representative_assemblies.txt",
          'r') as f:
    representative_structures = f.read().splitlines()

# Assembly info is contained in line index 16 to third last
representative_assemblies = representative_structures[16:-3]
all_assemblies = directory_structures[16:-3]

# Remove lines that only contain directory structure above assembly version
representative_assemblies = [l.split()[-1] for l in representative_assemblies if
                             l.split()[-1].split('/')[-1][:3] == 'GCF']
all_assemblies = [line.split()[-1] for line in all_assemblies if
                  line.split()[-1].split('/')[-1][:3] == 'GCF']

# Convert to data frames
representative_df = pd.DataFrame(
    [l.split('/') for l in representative_assemblies]).drop([1], 1)
assemblies_df = (pd.DataFrame([line.split('/') for line in all_assemblies])
                 .drop([1], 1))

# If there are more representative assemblies for one bacterium, pick either one
representative_df.drop_duplicates(subset=[0], inplace=True)

# Filter away bacteria that only have one assembly
single_assembly_bacteria = assemblies_df[
    (assemblies_df.groupby([0]).transform('count') == 1).values]
assemblies_df = assemblies_df[
    (assemblies_df.groupby([0]).transform('count') > 1).values]

# Combine assemblies with the representative assemblies
represented = pd.merge(assemblies_df.rename(columns={2: 'assembly'}),
                       representative_df.rename(columns={2: 'representative'}),
                       on=[0])

# Select the representative assembly
represented = (represented.query('assembly == representative')
               .drop(['representative'], 1))

# Select bacteria where there is no representative assemlby
not_represented = (assemblies_df[~assemblies_df[0].isin(representative_df[0])]
                   .copy())

# Isolate the version float of each assembly
not_represented['version'] = not_represented[2].str.extract(
    r"GCF_(\d+\.?\d?)_").astype(float)

# Take the newest assembly version for each bacterium
not_represented = not_represented.loc[
    not_represented.groupby(0)['version'].idxmax()].drop(['version'], 1)

# Join data frames to get a full list of those assemblies to keep
full_df = pd.concat(
    [represented, not_represented.rename(columns={2: 'assembly'})])

# Assert that bacteria with multiple assemblies have chosen an assembly
assert full_df[0].isin(assemblies_df[0]).all()

# Concatenate the single assembly bacteria
full_df = full_df.append(
    single_assembly_bacteria.rename(columns={2: 'assembly'}))

# Make sure that there is only one assembly for each bacterium
assert not full_df[0].duplicated().any()

# Asser that all bacteria have an assembly
assert assemblies_df[0].isin(full_df[0]).all()

full_df['filename'] = full_df['assembly'] + "_protein.faa.gz"

file_list = pd.Series(
    os.listdir("/home/wogie/Desktop/all_refseq_bac_sequences/"))

to_delete = file_list[~file_list.isin(full_df['filename'])]

to_delete = (
        "/home/wogie/Desktop/all_refseq_bac_sequences/" + to_delete).tolist()

DO_DELETE = True
if DO_DELETE:
    answer = input(f"Are you sure you want to delete {len(to_delete)} files\n"
                   f"{to_delete[:1]} and more files will be deleted")

    if answer == "yes":
        print("Deleting files...")
        [os.remove(p) for p in to_delete]

    print('Remaining files:', sum([os.path.exists(p) for p in to_delete]))
