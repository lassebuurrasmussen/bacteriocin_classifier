import gc
import multiprocessing
import os
from itertools import chain

import joblib

from code_modules.data_preprocessing.preprocessing_functions import \
    process_refseq_assembly_file

os.listdir("data/refseq/")

# Set path and get all files
# path = "/home/wogie/Desktop/refseq/all_refseq_bac_sequences/"
path = "data/refseq/all_refseq_bac_sequences/"
assemblies = os.listdir(path)

# Do extraction in batches to save memory
batch_size = 5000
for I, batch_start in enumerate(range(0, len(assemblies), batch_size)):
    print(f"Batch {I + 1} of {len(assemblies) // batch_size + 1}")

    # Add path to obtain full path for each file in the batch
    batch_assemblies = [path + a for a in
                        assemblies[batch_start:batch_start + batch_size]]

    # Run in parallel
    pool = multiprocessing.Pool(3)
    result = pool.starmap(process_refseq_assembly_file,
                          enumerate(batch_assemblies))

    # Unpack sequences and descriptions
    sequences = list(chain(*[r[0] for r in result]))
    descriptions = list(chain(*[r[1] for r in result]))

    print('Compressing files...')
    dump_path = "data/refseq/processed_refseq/"
    for variable, contents in zip(
            [batch_assemblies, sequences, descriptions],
            ["assembly_list", "all_sequences", "all_descriptions"]):
        joblib.dump(variable, f"{dump_path}{contents}_batch{I}.dump",
                    compress=0 if variable == "assembly_list" else 9)

    # Free up memory for next iteration
    del pool, result, sequences, descriptions
    gc.collect()
