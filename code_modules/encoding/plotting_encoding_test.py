import os

import gc
import time

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV

import code_modules.encoding.aa_encoding_functions as enc
import code_modules.encoding.encoding_testing_functions as enct

matplotlib.use('module://backend_interagg')  # Allennlp changes backend


def load_trained_model(model_path):
    # Load trained model
    grids: GridSearchCV = joblib.load(model_path)

    # Extract result
    in_cvresult = pd.DataFrame()
    for t, g in grids.items():
        dfi = pd.DataFrame(g.cv_results_)
        dfi['param_encode'] = t
        dfi['transformer'] = g
        in_cvresult = in_cvresult.append(dfi)

    in_cvresult.reset_index(drop=True, inplace=True)

    # Remove split scores and times
    in_cvresult = in_cvresult.loc[
                  :, in_cvresult.columns.str.find("split") == -1]
    in_cvresult = in_cvresult.loc[:, in_cvresult.columns.str.find("time") == -1]

    in_cvresult.drop(columns=['params', 'rank_test_score'], inplace=True)
    return in_cvresult


def plot_cv_results(in_cvresult, mname, save_fig=False):
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=[16, 9])

    # Extract scores of best alpha parameter and drop training scores
    in_cvresult = in_cvresult.copy()
    in_cvresult['best_alpha'] = (
        (in_cvresult.set_index('param_alpha')
         .groupby('param_encode')['mean_test_score']
         .transform('idxmax')
         .rename('best_alpha')
         .reset_index(drop=True)))

    in_cvresult = (in_cvresult
                   .query("param_alpha == best_alpha")
                   .drop(columns=['param_alpha', 'best_alpha',
                                  'mean_train_score', 'std_train_score']))

    for name_old, name_new in [('atchley_cluster', 'Atchley clust.'),
                               ('atchley', 'Atchley'),
                               ('onehot', 'One-Hot'),
                               ('reduced_alphabet', 'Reduced Alphabet'),
                               ('word_embedding_cluster', 'Word2Vec Clust.'),
                               ('word_embedding', 'Word2Vec'),
                               ('elmo_embedding_summed', 'ELMo summed'),
                               ('elmo_embedding', 'ELMo')]:
        in_cvresult['param_encode'].replace(name_old, name_new, inplace=True)

    in_cvresult = in_cvresult.loc[[3, 0, 14, 13, 4, 6, 11, 8]]

    sns.set()
    sns.reset_defaults()

    plt.close('all')
    fig, ax = plt.subplots(figsize=[10, 7])
    ax: plt.Axes
    plt.plot('param_encode', 'mean_test_score', 'b.', markersize=25,
             data=in_cvresult)
    plt.errorbar('param_encode', 'mean_test_score', 'std_test_score',
                 linewidth=4, data=in_cvresult, capsize=10, capthick=4)
    plt.xticks(rotation=10)
    ax.grid()
    ax.set_yticks(np.arange(round(ax.get_ylim()[0], 2),
                            round(ax.get_ylim()[1], 2) + 0.01, 0.01))
    ax.set_xlabel('Encoding')
    ax.set_ylabel('Accuracy')

    [ax.text(x - 0.4, y_, f"{s:.3f}")
     for x, y_, s in
     zip(range(len(in_cvresult['param_encode'])), in_cvresult.mean_test_score,
         in_cvresult.mean_test_score)]

    if save_fig:
        fig.tight_layout()
        fig.savefig(f"paper/figures/CV_score_{mname}")

    else:
        plt.show()


def get_test_performance(in_data, in_y, in_y_test, in_cvresult,
                         random_state, in_x,
                         in_x_test):
    best_idxs = in_cvresult.groupby('param_encode')['mean_test_score'].idxmax()

    clf_param_cols = in_cvresult.columns.str.extract('(param_.*)'). \
        dropna()[0].tolist()

    best_models = in_cvresult.loc[best_idxs, clf_param_cols +
                                  ['mean_test_score', 'transformer']]

    true_label_classifier = best_models.transformer.iloc[0].classes_[1]
    map_1_0 = pd.Series([0, 1], index=in_data.type.value_counts().index)

    if true_label_classifier == map_1_0[map_1_0 == 1].index:
        decision_val_inverter = 1
    else:
        print("+++Classifier true label different from least prevalent "
              "label+++")
        decision_val_inverter = -1
    in_y_test_1_0 = in_y_test.map(map_1_0)

    score_dict = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
                  'confusion_matrix': [], 'decision_val': []}

    for i, model in enumerate(best_models.index):
        print(f'Training {best_models.param_encode.loc[model]} {i + 1} of '
              f'{len(best_models.index)}')

        x_transformed, x_test_transformed = \
            in_x[best_models.param_encode.loc[model]], \
            in_x_test[best_models.param_encode.loc[model]]

        clf = best_models.loc[model, 'transformer'].best_estimator_

        np.random.seed(random_state)

        if best_models.param_encode.loc[model] == 'elmo_embedding':
            # Clean up to save memory and avoid crash
            gc.collect()
            time.sleep(2)

        clf.fit(x_transformed, in_y)

        y_test_pred = pd.Series(clf.predict(x_test_transformed)).map(map_1_0)

        for k, v in zip(list(score_dict.keys()),
                        [clf.score(x_test_transformed, in_y_test),
                         metrics.precision_score(in_y_test_1_0, y_test_pred),
                         metrics.recall_score(in_y_test_1_0, y_test_pred),
                         metrics.f1_score(in_y_test_1_0, y_test_pred),
                         metrics.confusion_matrix(in_y_test_1_0, y_test_pred) /
                         len(in_y_test) * 100,
                         clf.decision_function(
                             x_test_transformed) * decision_val_inverter
                         ]):
            score_dict[k].append(v)

    return score_dict, best_models, in_y_test_1_0


def plot_test_performance(best_models, score_dict, in_y_test_1_0, mname,
                          save_fig=False):
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=[16, 9])

    col_array = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', '#c04e01'])

    for name_old, name_new in [('atchley_cluster', 'Atchley clust.'),
                               ('atchley', 'Atchley'),
                               ('onehot', 'One-Hot'),
                               ('reduced_alphabet', 'Reduced Alphabet'),
                               ('word_embedding_cluster', 'Word2Vec Clust.'),
                               ('word_embedding', 'Word2Vec'),
                               ('elmo_embedding_summed', 'ELMo summed'),
                               ('elmo_embedding', 'ELMo')]:
        best_models['param_encode'].replace(name_old, name_new, inplace=True)

    # Precision-recall plots
    ax: plt.Axes = axes[0]
    for i, method in enumerate(best_models["param_encode"]):
        y_score = score_dict['decision_val'][i]

        precision, recall, thresh = precision_recall_curve(in_y_test_1_0,
                                                           y_score)

        ax.step(recall, precision,
                color=col_array[i],
                where='post',
                label=method)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(0.0, 1.0)
        ax.set_title('Precision-Recall curve')

    ax.legend()
    ax.grid()

    ax.set_yticks(np.arange(0, 1 + 0.05, 0.05))
    ax.set_xticks(np.arange(0, 1 + 0.05, 0.05))

    ax = axes[1]

    df_plot = pd.DataFrame(score_dict,
                           index=best_models.param_encode).reset_index()

    for name_old, name_new in [('atchley_cluster', 'Atchley clust.'),
                               ('atchley', 'Atchley'),
                               ('onehot', 'One-Hot'),
                               ('reduced_alphabet', 'Reduced Alphabet'),
                               ('word_embedding_cluster', 'Word2Vec Clust.'),
                               ('word_embedding', 'Word2Vec'),
                               ('elmo_embedding_summed', 'ELMo summed'),
                               ('elmo_embedding', 'ELMo')]:
        df_plot['param_encode'].replace(name_old, name_new, inplace=True)

    ax: plt.Axes
    # ax.plot('param_encode', 'accuracy', 'b.', markersize=25,
    #         data=df_plot)
    plt.plot('param_encode', 'accuracy', linewidth=4, data=df_plot, zorder=0)

    ax.scatter('param_encode', 'accuracy', s=500,
               data=df_plot, c=col_array)

    plt.xticks(rotation=15)
    ax.grid()
    ax.set_yticks(np.arange(round(ax.get_ylim()[0], 2),
                            round(ax.get_ylim()[1], 2) + 0.01, 0.01))
    ax.set_xlabel('Encoding')
    ax.set_ylabel('Accuracy')
    ax.set_title("Classification Accuracy Plot")

    [ax.text(x - 0.2, y_ + 0.005, f"{s:.3f}")
     for x, y_, s in zip(range(len(df_plot.index)), df_plot.accuracy,
                         df_plot.accuracy)]

    fig: plt.Figure

    if save_fig:
        fig.tight_layout()
        fig.savefig(f"paper/figures/test_set_score_{mname}")
    else:
        plt.show()


if __name__ == '__main__':
    datasets = ['GLYLIP', 'DNA_Rec_Int', 'halo', 'ANIPLA']
    use_mmseqs_cluster = True
    y, y_test, data, data_seq, X = enct.load_data(
        datasets[0], use_mmseqs_cluster=use_mmseqs_cluster)
    use_mmseqs_cluster = 'mmseqs' if use_mmseqs_cluster else ""
    maxlen = data['length'].max()
    Mname = f'{"_".join(sorted(data.type.unique()))}_{len(data)}' \
            f'{use_mmseqs_cluster}'
    model_results = load_trained_model(
        model_path=f'code_modules/saved_models/gridsearchmodel070319{Mname}')
    plot_cv_results(in_cvresult=model_results, mname=Mname, save_fig=True)

    # Get results on test set
    X_test = data_seq.loc[y_test.index]
    x_dump_location = f"data/uniprot/X_{Mname}"
    if (os.path.exists(f"{x_dump_location}.dump")
            and os.path.exists(f"{x_dump_location}_test.dump")):
        encoded_xs = joblib.load(f'{x_dump_location}.dump')
        encoded_xs_test = joblib.load(f'{x_dump_location}_test.dump')

        encoded_xs['word_embedding'] = encoded_xs['w2v_embedding']
        encoded_xs['word_embedding_cluster'] = encoded_xs[
            'w2v_embedding_cluster']

        del encoded_xs['w2v_embedding'], encoded_xs['w2v_embedding_cluster']

        encoded_xs_test['word_embedding'] = encoded_xs_test['w2v_embedding']
        encoded_xs_test['word_embedding_cluster'] = encoded_xs_test[
            'w2v_embedding_cluster']

        del encoded_xs_test['w2v_embedding'], encoded_xs_test[
            'w2v_embedding_cluster']

    else:
        encoded_xs = enc.get_all_enc(model_name=Mname, in_x=X, max_len=maxlen,
                                     dropfastdna=True)
        encoded_xs_test = enc.get_all_enc(model_name=Mname, in_x=X_test,
                                          max_len=maxlen, dropfastdna=True)
    test_performance_dict, top_models, y_test_1_0 = \
        get_test_performance(in_data=data, in_x=encoded_xs,
                             in_x_test=encoded_xs_test, in_y=y,
                             in_y_test=y_test, in_cvresult=model_results,
                             random_state=43)
    plot_test_performance(best_models=top_models,
                          score_dict=test_performance_dict,
                          in_y_test_1_0=y_test_1_0, mname=Mname, save_fig=True)
