import os
import time
from math import inf
from random import seed, shuffle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from pandas import Series
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, Dropout, Bidirectional, \
    LSTM, Convolution1D, MaxPooling1D, Flatten, Activation, \
    Input, Conv1D, \
    GlobalAveragePooling1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.utils import to_categorical

import code_modules.encoding.aa_encoding_functions as enc
import code_modules.encoding.encoding_testing_functions as enct


def elmo_encode_n_dump(from_dump=True, load_from_dump_path=None,
                       save_dump=False, model_name=None, max_len=None,
                       in_x=None, cuda_device=-1):
    if not from_dump:
        encoded_x = enc.elmo_embedding_encode(
            in_df=in_x, input_max_length=max_len, do_sum=True,
            cuda_device=cuda_device)

        if save_dump:
            dump_name = "_".join(model_name.split('_')[1:])
            path = f'code_modules/nn_training/{dump_name}/encoding_dumps/'
            dump_name = f'{path}x_train_elmo_encoded'

            i = 0
            while os.path.isfile(f'{dump_name}_n{i}'):
                i += 1
                if i > 100:
                    raise AssertionError

            dump_name = f'{dump_name}_n{i}'

            joblib.dump(encoded_x, filename=dump_name)

    else:
        encoded_x = joblib.load(load_from_dump_path)
        return encoded_x


def get_nn_model(x_shape, in_kwargs, architecture, use_tpu,
                 show_nn_summary=True):
    if in_kwargs is None:
        in_kwargs = {}

    available_architectures = {
        'BIDGRU': {'model_function': _bidgru, 'kwargs': {}},

        'DNN': {
            'model_function': _dnn,
            'kwargs': dict(l1_units=32, l2_units=32, dropout1=0.5,
                           dropout2=0.5, regularizer_vals=0.001)},

        'RNNCNN': {'model_function': _rnncnn,
                   'kwargs': dict(lr=0.001, conv_filters=320, kernel_size=26,
                                  pool_size=13, lstm_units=320, dropout1=0.2,
                                  dropout2=0.5)},

        'CNN': {'model_function': _cnn, 'kwargs': {}},

        'CNNPAR': {'model_function': _cnnpar,
                   'kwargs': dict(conv_filters1=50, kernel_sizes=None,
                                  maps_per_kernel=2, dropout1=0.1, dropout2=0.5,
                                  pool_size=3, conv_filters2=150,
                                  dense_units=60, lr=0.001)},

        'CNNPARLSTM': {'model_function': _cnnparlstm, 'kwargs': {}}
    }

    # Select architecture
    model_dict = available_architectures[architecture]

    # Get model parameters
    function_kwargs = model_dict['kwargs']
    function_kwargs.update({'x_shape': x_shape})  # Add shape parameter
    function_kwargs.update(in_kwargs)  # Update potential input parameters

    # Compile model
    if use_tpu:
        tpu_worker = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_worker)
        tf.contrib.distribute.initialize_tpu_system(resolver)
        strategy = tf.contrib.distribute.TPUStrategy(resolver)

        with strategy.scope():
            model = model_dict['model_function'](**function_kwargs)
    else:
        model = model_dict['model_function'](**function_kwargs)

    if show_nn_summary:
        print(x_shape)
        print(model.summary())

    return model


def _bidgru(x_shape):
    """https://github.com/nafizh/NeuBI/blob/master/neubi.py"""
    sequence_input = keras.layers.Input(shape=(x_shape[1], x_shape[2]),
                                        name='input_layer')

    x = keras.layers.Bidirectional(keras.layers.GRU(32, dropout=0.5,
                                                    recurrent_dropout=0.1,
                                                    return_sequences=True),
                                   name='blayer_1')(
        sequence_input)

    x = keras.layers.Bidirectional(keras.layers.GRU(32, dropout=0.5,
                                                    recurrent_dropout=0.1),
                                   name='blayer_2')(x)

    preds = keras.layers.Dense(1, activation='sigmoid', name='dlayer')(x)

    model = keras.models.Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['acc'])

    return model


def _dnn(x_shape, l1_units, l2_units, dropout1, dropout2, regularizer_vals):
    """https://github.com/lykaust15/Deep_learning_examples/blob/master/1.Fully
    _connected_psepssm_predict_enzyme/predict_enzyme.ipynb"""
    model = keras.models.Sequential([
        Dense(l1_units, activation='relu', input_shape=(x_shape[-1],),
              kernel_regularizer=keras.regularizers.l2(regularizer_vals)),
        Dropout(dropout1),
        Dense(l2_units, activation='relu',
              kernel_regularizer=keras.regularizers.l2(regularizer_vals)
              ),
        Dropout(dropout2),
        Dense(2, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['acc'])

    return model


def _rnncnn(x_shape, lr, conv_filters, kernel_size, pool_size, lstm_units,
            dropout1, dropout2):
    """https://github.com/lykaust15/Deep_learning_examples/blob/master/2.CNN_
    RNN_sequence_analysis/DNA_sequence_function_prediction.ipynb"""
    model = Sequential([
        Convolution1D(activation='relu', input_shape=(x_shape[1], x_shape[2]),
                      padding='valid',
                      filters=conv_filters, kernel_size=kernel_size,
                      ),

        MaxPooling1D(pool_size=pool_size, strides=13),
        Dropout(dropout1),

        Bidirectional(
            LSTM(lstm_units, return_sequences=True),
        ),
        Dropout(dropout2),
        Flatten(),

        Dense(input_dim=75 * 640, units=925),
        Activation('relu'),

        Dense(input_dim=925, units=1),
        Activation('sigmoid'),
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr),
                  metrics=['acc'])

    return model


def _cnn(x_shape):
    """https://github.com/lykaust15/Deep_learning_examples/blob/master/
    8.RBP_prediction_CNN/RBP_binding_site_prediction.ipynb"""
    model = Sequential([
        Conv1D(128, (10,), activation='relu',
               input_shape=(x_shape[1], x_shape[2])),
        Dropout(0.25),
        MaxPooling1D(pool_size=(3,), strides=(1,)),
        Conv1D(128, (10,), activation='relu', padding='same'),
        Dropout(0.25),
        MaxPooling1D(pool_size=(3,), strides=(1,)),
        Conv1D(256, (5,), activation='relu', padding='same'),
        Dropout(0.25),
        GlobalAveragePooling1D(),
        Dropout(0.25),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=Adadelta(),
                  metrics=['accuracy'])

    return model


def _cnnpar(x_shape, conv_filters1=50, kernel_sizes=None, maps_per_kernel=2,
            dropout1=0.1, dropout2=0.5, pool_size=3, conv_filters2=150,
            dense_units=60, lr=0.001):
    """https://github.com/emzodls/neuripp/blob/master/models.py"""
    if kernel_sizes is None:
        kernel_sizes = [3, 4, 5]

    inpt = Input(shape=(x_shape[1], x_shape[2]))
    kernel_sizes = kernel_sizes
    maps_per_kernel = maps_per_kernel
    convs = []

    for kernel_size in kernel_sizes:

        for map_n in range(maps_per_kernel):
            conv = Conv1D(filters=conv_filters1,
                          kernel_size=kernel_size,
                          padding='valid',
                          activation='relu',
                          kernel_initializer='glorot_normal',
                          strides=1)(inpt)
            conv_drop = Dropout(dropout1)(conv)
            max_pool = MaxPooling1D(pool_size)(conv_drop)
            convs.append(max_pool)

    merge = keras.layers.Concatenate(axis=1)(convs)
    mix = Conv1D(filters=conv_filters2,
                 kernel_size=kernel_sizes[0],
                 padding='valid',
                 activation='relu',
                 kernel_initializer='glorot_normal',
                 strides=1)(merge)
    max_pool = MaxPooling1D(3)(mix)
    flatten = Flatten()(max_pool)
    dense = Dense(dense_units, activation='relu')(flatten)
    drop = Dropout(dropout2)(dense)
    output = Dense(2, activation='sigmoid')(drop)
    model = keras.Model(inputs=inpt, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr),
                  metrics=['accuracy'])

    return model


def _cnnparlstm(x_shape):
    """https://github.com/emzodls/neuripp/blob/master/models.py"""
    inpt = keras.layers.Input(shape=(x_shape[1], x_shape[2]))
    kernel_sizes = [3, 4, 5]
    maps_per_kernel = 2
    convs = []

    for kernel_size in kernel_sizes:

        for map_n in range(maps_per_kernel):
            conv = keras.layers.Conv1D(filters=50,
                                       kernel_size=kernel_size,
                                       padding='valid',
                                       activation='relu',
                                       kernel_initializer='glorot_normal',
                                       strides=1)(inpt)
            conv_drop = keras.layers.Dropout(0.1)(conv)
            max_pool = keras.layers.MaxPooling1D(3)(conv_drop)
            convs.append(max_pool)
    merge = keras.layers.Concatenate(axis=1)(convs)
    mix = keras.layers.Conv1D(filters=150,
                              kernel_size=3,
                              padding='valid',
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              strides=1)(merge)
    max_pool = keras.layers.MaxPooling1D(3)(mix)
    lstm = keras.layers.Bidirectional(
        keras.layers.LSTM(50, return_sequences=False,
                          dropout=0.15, recurrent_dropout=0.15,
                          implementation=0))(max_pool)
    dense = keras.layers.Dense(60, activation='relu')(lstm)
    drop = keras.layers.Dropout(0.5)(dense)
    output = keras.layers.Dense(2, activation='sigmoid')(drop)
    model = Model(inputs=inpt, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def generate_parameter_grid(param_dict, n_permutations, kwarg_key_list):
    if kwarg_key_list is None:
        kwarg_key_list = ['l1_units', 'l2_units', 'dropout1', 'dropout2',
                          'regularizer_vals']

    parameter_types, parameter_choices = list(
        zip(*[[k, v] for k, v in param_dict.items()]))

    seed(573622)
    combinations_sample = []
    for choice_list in parameter_choices:

        choice_repeats = []
        i = 0
        while len(choice_repeats) < n_permutations:
            choice_repeats.append(choice_list[i % len(choice_list)])
            i += 1

        shuffle(choice_repeats)

        combinations_sample.append(choice_repeats)

    combinations_sample = list(zip(*combinations_sample))

    total_combinations = np.prod([len(l) for l in parameter_choices])
    print(f'Picking {n_permutations} ouf of {total_combinations}'
          f' permutations')

    params = [
        {'in_kwargs': {k: v for k, v in zip(parameter_types, values) if
                       k in kwarg_key_list}}
        for values in
        combinations_sample]

    epochs = [c[-2] for c in combinations_sample]
    batch_sizes = [c[-1] for c in combinations_sample]

    for p, e, bs in zip(params, epochs, batch_sizes):
        p.update(dict(epochs=e, batch_size=bs))

    return params


def run_cross_validation(nn_architechture: str, in_kwargs=None, n_folds=10,
                         do_elmo=True, do_w2v=False, epochs=100,
                         batch_size=1024, dataset_to_use='hamid',
                         save_logs=True, return_evaluation=False,
                         fold_i_to_skip=tuple(), use_tpu=False):
    """
    :param use_tpu:
    :param fold_i_to_skip:
    :param return_evaluation:
    :param save_logs:
    :param in_kwargs:
    :param do_elmo:
    :param nn_architechture: 'BIDGRU', 'DNN', 'CNN', 'CNNPAR', 'CNNPARLSTM' or
    'RNNCNN'
    :param n_folds:
    :param do_w2v:
    :param epochs:
    :param batch_size:
    :param dataset_to_use: 'hamid' or 'GLYLIP'
    """
    # Load data sets
    if dataset_to_use == 'hamid':
        kwargs = dict(dataset_to_use='hamid', use_mmseqs_cluster=False,
                      max_length=inf, min_length=0)
    elif dataset_to_use == 'GLYLIP':
        kwargs = dict(dataset_to_use='GLYLIP', use_mmseqs_cluster=True)
    else:
        raise AssertionError

    y, y_test, data, data_seq, x_raw = enct.load_data(**kwargs)

    # Set up model
    max_length = data['length'].max()
    flat_input = True if nn_architechture == 'DNN' else False
    output_is_categorical = True if nn_architechture == 'DNN' else False

    # Make unique model name
    model_name = (f'{nn_architechture}_{"_".join(sorted(data.type.unique()))}_'
                  f'len{len(data)}')

    x_w2v_encoded = enc.w2v_embedding_encode(in_df=x_raw,
                                             input_max_length=max_length)

    # Set up path for dumping pretrained ELMo embeddings
    elmo_dump_path = (f'code_modules/nn_training/'
                      f'{"_".join(model_name.split("_")[1:])}'
                      f'/encoding_dumps/x_train_elmo_encoded_n')
    if do_elmo:
        if not all(([f'{elmo_dump_path.split("/")[-1]}{i}'
                     in os.listdir("/".join(elmo_dump_path.split('/')[:-1]))
                     for i in range(n_folds)])):
            for f_i in range(n_folds):
                start_time = time.time()
                print(f"ELMo encoding {f_i}")
                elmo_encode_n_dump(from_dump=False, save_dump=True,
                                   model_name=model_name, max_len=max_length,
                                   in_x=x_raw, cuda_device=0)
                end_time = time.time()
                print(end_time - start_time)
        else:
            print("ELMo already encoded")

    # Transform y values to binary, with most frequent value as 0
    y_mapper = {k: v for v, k in enumerate(y.value_counts().index)}
    y_int = y.map(y_mapper)

    # Stratify K folds by y value ratios
    kfold = StratifiedKFold(n_splits=n_folds)
    folds = list(kfold.split(x_raw, y_int.values))

    # Specify which encodings to use
    encoding_styles = []
    encoding_styles.append('w2v') if do_w2v else None
    encoding_styles.append('elmo') if do_elmo else None

    # Specify their dimensions
    encoding_dimensions = []
    encoding_dimensions.append(200) if do_w2v else None
    encoding_dimensions.append(1024) if do_elmo else None

    # Run Cross Validation
    for encoding, embedding_size in zip(encoding_styles, encoding_dimensions):

        for fold_i, fold in enumerate(folds):
            if fold_i in fold_i_to_skip:
                continue

            tensorboard_entry_name = (f'{encoding}_{model_name}_'
                                      f'{int(time.time())}')

            # Select encoding method
            if encoding == 'w2v':
                x = x_w2v_encoded
            else:
                x = elmo_encode_n_dump(
                    load_from_dump_path=f"{elmo_dump_path}{fold_i}")

            if not flat_input:
                # Convert each AA to an embedding matrix
                x = x.reshape(len(x), -1, embedding_size)

            x_fold_train = x[fold[0]]
            x_fold_val = x[fold[1]]

            y_fold_train = y_int.iloc[fold[0]]
            y_fold_val = y_int.iloc[fold[1]]

            # Make a unique log for each run
            tensorboard_entry_name_i = f"{tensorboard_entry_name}_fold{fold_i}"

            tensorboard = TensorBoard(log_dir=f"code_modules/nn_training/logs/"
                                              f"{tensorboard_entry_name_i}")

            if output_is_categorical:
                # Convert to one-hot output instead of 0-1
                y_fold_train = to_categorical(y_fold_train)
                y_fold_val = to_categorical(y_fold_val)

            # Compile model
            nn_model = get_nn_model(x_shape=x.shape, in_kwargs=in_kwargs,
                                    architecture=nn_architechture,
                                    use_tpu=use_tpu)

            print(tensorboard_entry_name_i)

            callbacks = [tensorboard] if save_logs else None
            steps_per_epoch = 1 if use_tpu else None

            # Fit model
            nn_model.fit(x_fold_train, y_fold_train,
                         validation_data=(x_fold_val, y_fold_val),
                         epochs=epochs, callbacks=callbacks,
                         batch_size=batch_size, steps_per_epoch=steps_per_epoch)

            if return_evaluation:
                # Return the error and accuracy and abandon rest of folds
                return (nn_model.evaluate(x_fold_val, y_fold_val),
                        nn_model.evaluate(x_fold_train, y_fold_train))

            # Delete model to clean up memory
            del nn_model


def hyperparameter_gridsearch(param_dict, nn_architechture, i_to_skip=tuple(),
                              n_permutations=None, kwarg_key_list=None):
    # TODO - this method creates duplicates!!!!
    params = generate_parameter_grid(param_dict=param_dict,
                                     n_permutations=n_permutations,
                                     kwarg_key_list=kwarg_key_list)

    time_stamp = int(time.time())
    fname = f"code_modules/nn_training/temp_log{time_stamp}.txt"

    # Create new log
    log_file = open(fname, 'w')
    log_file.close()

    result_list = []
    for p_i, p_dict in enumerate(params):
        if p_i in i_to_skip:
            continue

        print("Grid search parameters:")
        print(p_dict)

        result_i = run_cross_validation(nn_architechture=nn_architechture,
                                        save_logs=False,
                                        return_evaluation=True,
                                        **p_dict
                                        )
        result_list.append(result_i)

        # Save to log
        d = p_dict.copy()
        d.update(dict(result=result_i, p_i=p_i))
        with open(fname, 'a') as f:
            f.write(str(d) + "\n")

    return result_list


def encode_elmo_dump(x_raw, data, dump_path):
    if os.path.exists(dump_path):
        print(f"Loading pretrained ELMo...")
        encoded_x = joblib.load(dump_path)

        return encoded_x

    else:
        print(f"Encoding ELMo...")
        max_length = data['length'].max()
        encoded_x = enc.elmo_embedding_encode(
            in_df=x_raw, input_max_length=max_length, do_sum=True,
            cuda_device=0)
        joblib.dump(encoded_x, dump_path)

        return None


def train_final_model(nn_architechture='CNNPAR', in_kwargs=None, epochs=68,
                      batch_size=1024, just_test_model=False,
                      weights_path=None, show_nn_summary=False):
    if in_kwargs is None:
        # Choose best hyper parameters
        in_kwargs = {'conv_filters1': 75, 'conv_filters2': 300,
                     'dense_units': 120, 'dropout1': 0.0, 'dropout2': 0.0,
                     'kernel_sizes': [6, 8, 10], 'lr': 0.001,
                     'maps_per_kernel': 2, 'pool_size': 3}

    # Load data
    y, y_test, data, data_seq, x_raw = enct.load_data(
        dataset_to_use='hamid', use_mmseqs_cluster=False, max_length=inf,
        min_length=0)
    x_test = data_seq.loc[y_test.index]

    x_type = 'test' if just_test_model else 'train'
    dump_path = (f"code_modules/nn_training/BAC_UNI_len2006/encoding_dumps/"
                 f"x_{x_type}_elmo_encoded_final")

    if just_test_model:
        encoded_x = encode_elmo_dump(x_raw=x_test, data=data,
                                     dump_path=dump_path)
    else:
        encoded_x = encode_elmo_dump(x_raw=x_raw, data=data,
                                     dump_path=dump_path)
    if encoded_x is None:
        print("Shutting down")
        return

    # Transform y values to binary, with most frequent value as 0
    # y_mapper = {k: v for v, k in enumerate(y.value_counts().index)}
    y_mapper = {'UNI': 0, 'BAC': 1}  # This has been verified

    if just_test_model:
        y_int = y_test.map(y_mapper)

    else:
        y_int = y.map(y_mapper)

    # Specify embedding dimension
    encoding_dimension = 1024

    # Convert each AA to an embedding matrix
    encoded_x = encoded_x.reshape(len(encoded_x), -1, encoding_dimension)

    if just_test_model:
        # Assert that test data set has been chosen
        assert len(encoded_x) == len(x_test)
        assert len(y_int) == len(y_test)

    # Set up and compile model
    nn_model = get_nn_model(x_shape=encoded_x.shape, in_kwargs=in_kwargs,
                            architecture=nn_architechture, use_tpu=False,
                            show_nn_summary=show_nn_summary)

    if just_test_model:
        nn_model.load_weights(weights_path)

        evaluate_test_results(nn_model=nn_model, x_test=encoded_x, y_test=y_int)

        return

        # Make folder for saving model weights
    model_path = "code_modules/nn_training/saved_models/"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # Make unique model name
    model_path = (f'{model_path}final_elmo_{nn_architechture}'
                  f'_{"_".join(sorted(data.type.unique()))}_len{len(data)}'
                  f'_{int(time.time())}')

    print(model_path)

    # Fit model
    nn_model.fit(encoded_x, y_int, epochs=epochs, batch_size=batch_size)

    # Save weights
    nn_model.save_weights(model_path)

    # Load model and validate that it's correctly saved
    del nn_model
    nn_model = get_nn_model(x_shape=encoded_x.shape, in_kwargs=in_kwargs,
                            architecture=nn_architechture,
                            use_tpu=False)
    nn_model.evaluate(encoded_x, y_int)  # Untrained model
    nn_model.load_weights(model_path)
    nn_model.evaluate(encoded_x, y_int)  # Trained model


def evaluate_test_results(nn_model: keras.Model, x_test: np.ndarray,
                          y_test: Series, get_difficult_cases=False):
    # Make predictions on test set
    predictions = nn_model.predict(x_test)

    # Function that evalutaes accuracy
    def do_e(preds, ys, boolean_indexer=None):
        if boolean_indexer is None:
            return np.mean(preds.argmax(1).flatten() == ys)

        else:
            return np.mean(preds[boolean_indexer].argmax(1).flatten()
                           == ys[boolean_indexer])

    # Calculate accuracies given varying threshold certainties
    certainties = np.linspace(0, 1, 1000)
    accuracies = []
    for certainty in certainties:
        where = predictions.sum(1) > certainty
        accuracies.append(do_e(predictions, y_test, where))

    # Plot accuracy and density of certainties
    def y_accuracy_plot():
        fig: plt.Figure = plt.figure()

        # Accuracy plot
        ax1: plt.Axes = plt.axes()
        ax1.plot(certainties, accuracies, 'b', label='accuracy')

        ax1.set_xticks(np.arange(0, 1.1, 0.1))
        ax1.grid()
        ax1.set_title(f"Mean test accuracy: {do_e(predictions, y_test):.3f}")
        ax1.set_ylabel("Accuracy")
        ax1.set_xlabel("Certainty")

        # Density plot
        twinax: plt.Axes = ax1.twinx()
        sns.kdeplot(predictions.max(1), ax=twinax, label='certainty density',
                    color='red')
        twinax.legend().remove()
        twinax.set_ylim(top=15)

        twinax.set_ylabel("Density")
        twinax.set_yticks([])

        fig.legend(loc=[0.2, 0.8])
        fig.tight_layout()
        plt.savefig("code_modules/nn_training/BAC_UNI_len2006/test_accuracy")

    y_accuracy_plot()

    if get_difficult_cases:
        difficult_cases_idxs = y_test.iloc[predictions.max(1) < 0.3].index
        return difficult_cases_idxs
