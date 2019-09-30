import gc
import time

import joblib
import numpy as np
import pandas as pd

import os

os.listdir("data/refseq/processed_refseq")

for batch_i in range(5):
    print(f"Batch {batch_i + 1} of 5")

    # Load in joblib dump
    start_time = time.time()
    sequences = joblib.load(
        f"data/refseq/processed_refseq/all_sequences_batch{batch_i}.dump")
    end_time = time.time()
    print(end_time - start_time)
    gc.collect()

    # Convert series
    print("Removing duplicates")
    sequence_df = pd.Series(sequences)
    del sequences  # remove old object to clear memory
    gc.collect()

    # Check if the sequence without assembly number is duplicated. If so, remove
    not_duplicates = ~sequence_df.str[5:].duplicated()
    len_before_removal = len(sequence_df)  # Original row count
    sequence_df = sequence_df[not_duplicates]
    gc.collect()

    # Save sequences, assemblies that they belong to and the indices removed to
    # hard disk
    print("Compressing sequences")
    dump_path = "data/refseq/processed_refseq/"
    fname = f"all_sequences_processing_step2_batch{batch_i}.dump"
    joblib.dump(sequence_df.str[5:].tolist(),
                f"{dump_path}{fname}", compress=9)
    gc.collect()

    print("Saving assembly belongings")
    fname = f"all_sequences_assemblies_processing_step2_batch{batch_i}.csv"
    (sequence_df.str[:5].value_counts().sort_index().cumsum().
     to_csv(f"{dump_path}{fname}", header=True))
    gc.collect()

    fname = f"indices_removed_step2_batch{batch_i}.dump"
    missing = np.arange(len_before_removal)
    missing = missing[~pd.Series(missing).isin(sequence_df.index)]
    joblib.dump(missing.tolist(), f"{dump_path}{fname}")

    # Clear memory for next iteration
    del sequence_df
    gc.collect()
