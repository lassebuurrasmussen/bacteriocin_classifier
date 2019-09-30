###############################create minibatches###############################
import joblib

MINIBATCH_SIZE = 10_000

for batch in range(5):
    print(f"Batch {batch + 1} of 5")

    path = "data/refseq/processed_refseq/"
    batch_patch = f"all_sequences_processing_step2_batch{batch}.dump"

    sequences = joblib.load(path + batch_patch)

    minibatch_start_idxs = list(range(0, len(sequences), MINIBATCH_SIZE))
    path += "minibatches/"
    for i, minibatch_start_i in enumerate(minibatch_start_idxs):
        if not i % 1000:
            print(f"Minibatch {i + 1} of {len(minibatch_start_idxs)}")

        joblib.dump(
            sequences[minibatch_start_i:minibatch_start_i + MINIBATCH_SIZE],
            f"{path}batch{batch}minibatch_{i}")

#########################create minibatches in parallel#########################
# import multiprocessing
# from itertools import repeat
#
# import code_modules.data_preprocessing.preprocessing_functions as pfncs
#
# MINIBATCH_SIZE = 100_000
#
# pool = multiprocessing.Pool(5)
#
# parameter_iterable = list(zip(range(5), list(repeat(MINIBATCH_SIZE, 5))))
#
# pool.starmap(pfncs.process_seq_batch, parameter_iterable)
