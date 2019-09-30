import importlib
import os
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

import code_modules.encoding.aa_encoding_functions as enc
import code_modules.nn_training.functions as fncs

# Activate GPU
tf.test.gpu_device_name()

importlib.reload(enc)

CUDA_DEVICE = 0
# CUDA_DEVICE = 1
MAX_LENGTH = 359
PATH_TO_MINIBATCHES = "data/refseq/processed_refseq/minibatches/"
WEIGHTS_PATH = ("code_modules/nn_training/BAC_UNI_len2006/final_elmo_CNNPAR"
                "_BAC_UNI_len2006/final_elmo_CNNPAR_BAC_UNI_len2006"
                "_1565079880")
IN_KWARGS = {'conv_filters1': 75, 'conv_filters2': 300,
             'dense_units': 120, 'dropout1': 0.0, 'dropout2': 0.0,
             'kernel_sizes': [6, 8, 10], 'lr': 0.001,
             'maps_per_kernel': 2, 'pool_size': 3}

minibatches = sorted(os.listdir(PATH_TO_MINIBATCHES))
I_SKIP_ITER = []
existing_results = os.listdir("code_modules/nn_training/application/results")

for i, minibatch in enumerate(minibatches):
    batch_n = minibatch[5]
    minibatch_n = minibatch.split("_")[-1]

    print(f"Minibatch {minibatch_n} ({i + 1} of {len(minibatches)})")

    if i in I_SKIP_ITER:
        continue

    result_fname = f"batch{batch_n}_minibatch{minibatch_n}"
    if result_fname in existing_results:
        print("Skipping as it already exists..")
        continue

    sequences = joblib.load(PATH_TO_MINIBATCHES + minibatch)
    sequences = pd.Series(sequences)

    print("Embedding".center(80, "#"))
    start_time = time.time()
    encoded = enc.elmo_embedding_encode(enc.expand_seqstr_series(sequences),
                                        input_max_length=MAX_LENGTH,
                                        do_sum=True,
                                        cuda_device=CUDA_DEVICE)
    end_time = time.time()
    print(end_time - start_time)

    # # Clear CPU memory
    # gc.collect()
    # cuda.select_device(0)
    # cuda.close()
    # cuda.select_device(1)
    # cuda.close()

    # Specify embedding dimension
    encoding_dimension = 1024

    # Convert each AA to an embedding matrix
    encoded = encoded.reshape(len(encoded), -1, encoding_dimension)

    print("Making predictions".center(80, "#"))
    # Set up TF session without GPU
    start_time = time.time()
    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(config=config) as sess:
        # Set up and compile model
        nn_model = fncs.get_nn_model(x_shape=encoded.shape, in_kwargs=IN_KWARGS,
                                     architecture="CNNPAR", use_tpu=False,
                                     show_nn_summary=True)
        nn_model.load_weights(WEIGHTS_PATH)

        # Make predictions on test set
        predictions = nn_model.predict(encoded)

    end_time = time.time()
    print(end_time - start_time)

    # Take only those predicted to be bacteriocins
    bac_predictions_indices = predictions.argmax(1) == 1
    bac_predictions = predictions[bac_predictions_indices]

    # Save the probabilities along with indices
    bac_predictions_indices = np.where(bac_predictions_indices)[0]
    save_file = (np.c_[bac_predictions_indices, bac_predictions].round(4)
                 .tolist())

    # Save results
    out_dir = "code_modules/nn_training/application/results/"
    joblib.dump(save_file, out_dir + result_fname)
