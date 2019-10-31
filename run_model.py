import pandas as pd
import tensorflow as tf

import code_modules.encoding.aa_encoding_functions as enc
import code_modules.nn_training.functions as fncs
from code_modules.data_preprocessing.preprocessing_functions import fasta2csv
import sys


def main(input_fasta, out_dir):
    if out_dir is None:
        out_dir = "sample_result.txt"

    if input_fasta is None:
        input_fasta = "sample_fasta.faa"

    with open(input_fasta, 'r') as f:
        fasta = f.read()

    # Conver input fasta to csv
    df = fasta2csv(input_fasta)

    # Activate GPU
    cuda_device = tf.test.gpu_device_name()
    cuda_device = -1 if cuda_device == '' else 0

    max_length = 359

    assert all(df['Sequence'].str.len() <= max_length), f"String length exceeding {max_length}"

    weights_path = ("code_modules/nn_training/BAC_UNI_len2006/final_elmo_CNNPAR_BAC_UNI_len2006"
                    "/final_elmo_CNNPAR_BAC_UNI_len2006_1565079880")
    in_kwargs = {'conv_filters1': 75, 'conv_filters2': 300,
                 'dense_units': 120, 'dropout1': 0.0, 'dropout2': 0.0,
                 'kernel_sizes': [6, 8, 10], 'lr': 0.001,
                 'maps_per_kernel': 2, 'pool_size': 3}

    sequences = pd.Series(df['Sequence'])

    encoded = enc.elmo_embedding_encode(enc.expand_seqstr_series(sequences),
                                        input_max_length=max_length,
                                        do_sum=True,
                                        cuda_device=cuda_device)

    # Specify embedding dimension
    encoding_dimension = 1024

    # Convert each AA to an embedding matrix
    encoded = encoded.reshape(len(encoded), -1, encoding_dimension)

    # Set up TF session without GPU
    config = tf.ConfigProto(device_count={"GPU": 0})
    with tf.Session(config=config) as sess:
        # Set up and compile model
        nn_model = fncs.get_nn_model(x_shape=encoded.shape, in_kwargs=in_kwargs,
                                     architecture="CNNPAR", use_tpu=False,
                                     show_nn_summary=True)
        nn_model.load_weights(weights_path)

        # Make predictions on test set
        predictions = nn_model.predict(encoded)

    out_df = df['Name']
    out_df = pd.concat([out_df,
                        pd.DataFrame(predictions, columns=['not_bacteriocin_score',
                                                           'bacteriocin_score'])],
                       axis=1)

    # Save results
    out_df.to_csv(out_dir)


if __name__ == '__main__':
    main(input_fasta=sys.argv[1], out_dir=sys.argv[2])
