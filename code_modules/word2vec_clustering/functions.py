import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from tqdm import tqdm

import code_modules.encoding.aa_encoding_functions as enc


def get_bacteriocins(n_each, camp_source, drop_non_bacteriocins,
                     drop_bagel_bactibase, type_by_camp_belonging,
                     upper_length_cutoff=255):
    # Get data
    paths_bacteriocins = ["data/bactibase/bactibase021419.csv",
                          "data/bagel/bagel021419.csv"]
    out_data = pd.concat([pd.read_csv(p, '\t', usecols=['ID', 'Sequence'])
                          for i, p in enumerate(paths_bacteriocins)], ignore_index=True)
    out_data['bacteriocin'] = 'bactibase/bagel'  # Add label
    out_data['class'] = (out_data['ID'].astype(str).str.split('.', expand=True).iloc[:, -1].
                         fillna(0))

    camp = pd.read_csv("data/camp/camp_database03-26-19.csv")

    if camp_source:
        camp = camp[camp['Source'].str.contains('[Ss]apiens', na=False)]

    # Label bacteriocins from CAMP
    camp['bacteriocin'] = np.array(['not_bacteriocin_CAMP', 'bacteriocin_CAMP'])[
        camp.apply(lambda col: col.astype(str).
                   str.contains('[Bb]acteriocin')).values.any(1).astype(int)]

    camp['class'] = np.where(camp['bacteriocin'] == 'bacteriocin_CAMP', 4, None)

    out_data = out_data.append(camp[['CAMP_ID', 'Sequence', 'bacteriocin', 'class']].
                               rename(columns={'CAMP_ID': 'ID'}), sort=False)
    out_data['class'] = out_data['class'].fillna(np.nan).astype(float)

    # Clean data
    out_data.drop_duplicates(subset='Sequence', inplace=True)
    out_data.dropna(subset=['Sequence'], inplace=True)
    out_data['Sequence'] = out_data['Sequence'].str.replace(' ', '')
    out_data['Sequence'] = out_data['Sequence'].str.replace(',.+?$', '')
    out_data = out_data[~out_data['Sequence'].str.contains('-')]

    out_data = out_data[~out_data['Sequence'].str.contains('[ZB]')]

    out_data['Sequence'] = out_data['Sequence'].str.upper()

    assert out_data['Sequence'].str.isupper().all()
    assert out_data['Sequence'].str.isalpha().all()

    length_bool = out_data['Sequence'].apply(len) <= upper_length_cutoff

    out_data: pd.DataFrame = out_data[length_bool].reset_index(drop=True)

    if drop_non_bacteriocins:
        out_data = out_data[out_data['bacteriocin'] != 'not_bacteriocin_CAMP']

    if drop_bagel_bactibase:
        out_data = out_data[out_data['bacteriocin'] != 'bactibase/bagel']

    if n_each:
        np.random.seed(3799321)
        idxs = np.concatenate([np.random.choice(
            out_data[out_data['bacteriocin'] == t].index, n_each,
            replace=False) for t in out_data['bacteriocin'].unique()])

        out_data = out_data.reindex(idxs)

    if type_by_camp_belonging:
        out_data.loc[out_data['bacteriocin'].str.contains('CAMP'),
                     'bacteriocin'] = 'CAMP'

    return out_data


def do_pca(in_array, n_components, copy=True, svd_solver='auto'):
    pca = PCA(n_components=n_components, copy=copy, svd_solver=svd_solver)
    fitted_data = pca.fit_transform(in_array)
    return fitted_data, pca.explained_variance_ratio_


def make_plot(xyarray, bacteriocin_series, pc_vars=None, kind='scatter',
              multiple_axes=False, custom_color_dict=None, title='',
              zoom=None, shade=False, shade_lowest=False, subset_list=None,
              show_plot=True, custom_axis_labels=None, s=6, alpha=0.5,
              axnrowscols=None, save_plot=False, save_path=None,
              figsize=(10, 7)):
    plt.close('all')

    ax, axes = None, None
    if multiple_axes:
        axnrowscols = [2, 2] if axnrowscols is None else axnrowscols
        f, axes = plt.subplots(*axnrowscols, figsize=figsize, sharex='all',
                               sharey='all')
        axes = axes.flatten()
    else:
        f, ax = plt.subplots(figsize=figsize)

    color_dict = ({'not_bacteriocin_CAMP': 'g', 'bactibase/bagel': 'r',
                   'bacteriocin_CAMP': 'b'} if custom_color_dict is None else
                  custom_color_dict)

    for i, bac_label in enumerate(color_dict.keys()):
        ax = axes[i] if multiple_axes else ax

        plot_array = xyarray[bacteriocin_series == bac_label]

        color = color_dict[bac_label] if not multiple_axes else 'blue'

        if kind == 'scatter':
            ax.plot(*plot_array.T, '.', markersize=s, color=color,
                    label=bac_label, alpha=alpha)

            if not multiple_axes and subset_list is not None:
                for i_, l in enumerate(subset_list):
                    [ax.text(-3.7 + i_ * 6, 10 - i, (
                        f'fraction of {k} in subset: {v:.0f}%'))
                     for i, (k, v) in enumerate(l.items())]

        elif kind == 'kde':
            sns.kdeplot(*plot_array.T, label=bac_label, ax=ax, shade=shade,
                        shade_lowest=shade_lowest)

        else:
            raise AssertionError

        ax.set_title(bac_label) if multiple_axes else None

        axlabs = (['PC1', 'PC2'] if custom_axis_labels is None else
                  custom_axis_labels)

        if pc_vars is not None:
            axlabs = [f'{axlabs[_i]} ({pc_vars[_i] * 100 :.2f}%)'
                      for _i in [0, 1]]

        ax.set_xlabel(axlabs[0])
        ax.set_ylabel(axlabs[1])

        if zoom is not None:
            ax.axis(zoom)

    plt.suptitle(title, x=0.5, y=0.995, fontsize=13, fontweight='bold')
    plt.tight_layout()
    ax.legend() if not multiple_axes else None
    if not save_plot:
        plt.show() if show_plot else None
    else:
        f.savefig(f'{save_path}{title.replace(" ", "_")}')


def _circle_y(r, xs, center):
    ys_upper = np.sqrt(r ** 2 - xs ** 2)
    ys_lower = -ys_upper

    return ys_upper + center[1], ys_lower + center[1]


def make_circle(center, r, in_ax=None, draw=True, in_x=None, in_y=None):
    n = 200
    xs = np.linspace(0 - r, 0 + r, n)

    ys_upper, ys_lower = _circle_y(r, xs, center)

    xs += center[0]

    if draw:
        in_ax.plot(xs, ys_upper, '-b')
        in_ax.plot(xs, ys_lower, '-b')

    if in_x is not None:
        # Find the xs within the circle
        x_in_circle = (r + center[0] >= in_x) & (-r + center[0] <= in_x)

        # Isolate these and remeber their indices
        x_subset = pd.Series(in_x)
        x_subset = x_subset[x_in_circle] - center[0]

        # Calculate the y value of the circle at the location of these xs
        circle_ys_upper, circle_ys_lower = _circle_y(r, x_subset.values, center)

        # Take the y values of the subset and remember indices
        y_subset = pd.Series(in_y[x_in_circle], index=x_subset.index)

        # Check whether these y values are below and above the respective
        # thresholds
        y_subset_in_circle = pd.Series((y_subset <= circle_ys_upper) &
                                       (y_subset >= circle_ys_lower))

        # Combine with the rest of the y values using the index
        y_in_circle = pd.Series(np.repeat(False, len(in_y)))
        y_in_circle.loc[y_subset_in_circle.index] = y_subset_in_circle

        return np.logical_and(y_in_circle, x_in_circle)


def load_uniprot(n_each):
    path = "data/uniprot/carbohydrate_metabolic_process_many_taxa.csv"
    out_df = pd.read_csv(path)

    if n_each:
        np.random.seed(3799321)
        idxs = np.concatenate([np.random.choice(
            out_df[out_df['type'] == t].index, n_each, replace=False)
            for t in out_df['type'].unique()])

        out_df = out_df.reindex(idxs)

    # Remove Z and return
    return out_df[~out_df['Sequence'].str.contains('Z')]


def load_and_combine_data(n_each_uniprot=135, n_each_bac=None, camp_source=None,
                          drop_non_bacteriocins=True,
                          drop_bagel_bactibase=False,
                          type_by_camp_belonging=False):
    """
    :param n_each_uniprot:
    :param n_each_bac:
    :param camp_source: str to only get camp of source containing [Ss]apiens
    :param drop_non_bacteriocins:
    :param drop_bagel_bactibase: True to remove bagel and bactibase
    :param type_by_camp_belonging: True to have type being CAMP if from CAMP
    :return:
    """
    # Load bacteriocin sequences
    data_bacteriocins = (get_bacteriocins(
        n_each=n_each_bac, camp_source=camp_source,
        drop_non_bacteriocins=drop_non_bacteriocins,
        drop_bagel_bactibase=drop_bagel_bactibase,
        type_by_camp_belonging=type_by_camp_belonging).
                         drop(columns=['class']).
                         rename(columns={'bacteriocin': 'type'}))

    # Load UniProt sequences
    data_uniprot = (load_uniprot(n_each=n_each_uniprot).rename(
        columns={'Entry': 'ID'}).drop(columns='Entry name'))

    # Combine data frames
    assert np.all(data_bacteriocins.columns == data_uniprot.columns)
    return pd.concat([data_bacteriocins, data_uniprot], ignore_index=True)


def make_color_dict(in_df, type_column='type'):
    types = in_df[type_column].unique()
    colors = ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000',
              '#95d0fc', '#029386', '#f97306', '#35063e',
              '#380282'][:len(types)]

    assert len(colors) == len(types)

    color_dict = {k: v for k, v in zip(types, colors)}

    return color_dict


def do_pls(in_x, in_y_binary, n_components=2, get_estimator=False,
           get_transformed=True):
    reg = MyPLSClassifier(n_components=n_components)
    reg.fit(in_x, in_y_binary)
    in_x_transformed = reg.transform(in_x) if get_transformed else None

    return_list = []
    if get_transformed:
        return_list.append(in_x_transformed)
    if get_estimator:
        return_list.append(reg)

    return tuple(return_list) if len(return_list) > 1 else return_list[0]


def loop_plotting(in_xs_transformed, in_explained_variances, in_ys,
                  in_ys_binary, in_color_dict, in_windows, do_binary='both',
                  t_type='PCA', color_by_bacteriocin=True):
    binary_options = [False, True] if do_binary == 'both' else [do_binary]

    binary_dict = ({'not bacteriocin': 'red', 'bacteriocin': 'blue'}
                   if color_by_bacteriocin else {'not CAMP': 'red',
                                                 'CAMP': 'blue'})

    for binary in tqdm(binary_options):
        y_loop_list = in_ys_binary if binary else in_ys
        c = (binary_dict if binary
             else in_color_dict)

        for w, x_transformed, explained_variance, y in zip(
                in_windows, in_xs_transformed, in_explained_variances,
                y_loop_list):
            make_plot(xyarray=x_transformed, bacteriocin_series=y,
                      pc_vars=explained_variance, custom_color_dict=c,
                      title=f'{t_type}_windowed w={w} binary={binary}',
                      show_plot=False, s=6, alpha=0.5, save_plot=True,
                      save_path=f"plots/sliding_window_plots/{t_type}_windowed_w{w}",
                      figsize=[10, 7])


def write_coef_ranking(in_coef, write_to_desk=False):
    # Ranked highest to lowest
    coef_ranking = abs(in_coef).argsort()[::-1]
    time_stamp = time.strftime('%d%h_%H-%M')

    if write_to_desk:
        with open(f"code_modules/word2vec_clustering/PLS_coeff_ranking_"
                  f"descending_{time_stamp}.txt", 'w') as f:
            f.write("\n".join(list(map(str, coef_ranking.tolist()))))

    return coef_ranking


def write_col_ranking(in_scores, number_of_cols, write_to_desk=False):
    time_stamp = time.strftime('%d%h_%H-%M')

    if write_to_desk:
        with open(f"code_modules/word2vec_clustering/PLS_col_ranking_"
                  f"{time_stamp}.txt", 'w') as f:
            f.write("\n".join(list(map(str, in_scores.tolist()))))

            f.write('\nNumber of cols:\n')
            f.write(",".join(list(map(str, number_of_cols))))


def load_ranking(path=None):
    if not path:
        path = ("code_modules/word2vec_clustering/PLS_coeff_ranking"
                "_descending_12Jun_16-11.txt")

    with open(path, 'r') as f:
        ranking = f.read().splitlines()

    return [int(r) for r in ranking]


def differing_window_plots(in_data, windows=(10, 30, 50), add_to_filenames='',
                           color_by_bacteriocin=True):
    """

    :param in_data:
    :param windows:
    :param add_to_filenames:
    :param color_by_bacteriocin: False to color by CAMP belonging
    :return:
    """
    xs, ys, ys_binary, xs_pca, xs_pls, explained_variances = ([], [], [], [],
                                                              [], [])

    for w in tqdm(windows):
        print('Encoding...')
        x, y = enc.w2v_embedding_window_encode(in_data, in_data['Sequence'], w=w)

        mapper = ({'bactibase/bagel': 1, 'bacteriocin_CAMP': 1}
                  if color_by_bacteriocin else {'CAMP': 1})

        mapper2 = ({1: 'bacteriocin', 0: 'not bacteriocin'}
                   if color_by_bacteriocin else {1: 'CAMP', 0: 'not CAMP'})

        y_binary = (y.map(mapper).fillna(0).
                    map(mapper2))

        print('Doing PCA...')
        # PCA
        x_pca, explained_variance = do_pca(x, n_components=2)

        print('Doing PLS...')
        # PLS
        x_pls = do_pls(x, y_binary.map({v: k for k, v in mapper2.items()}))

        [l.append(e) for l, e in zip([xs, ys, ys_binary, xs_pca, xs_pls, explained_variances],
                                     [x, y, y_binary, x_pca, x_pls, explained_variance])]

    color_dict = make_color_dict(in_data)

    loop_plotting(in_xs_transformed=xs_pca,
                  in_explained_variances=explained_variances, in_ys=ys,
                  in_ys_binary=ys_binary, in_color_dict=color_dict,
                  in_windows=windows, t_type=f'PCA{add_to_filenames}',
                  color_by_bacteriocin=color_by_bacteriocin)

    loop_plotting(in_xs_transformed=xs_pls,
                  in_explained_variances=[None] * 3, in_ys=ys,
                  in_ys_binary=ys_binary, in_color_dict=color_dict,
                  in_windows=windows, t_type=f'PLS{add_to_filenames}',
                  color_by_bacteriocin=color_by_bacteriocin)


class MyPLSClassifier(ClassifierMixin, PLSRegression):
    """Custom PLS class. Rounds the prediction to become a classifier and uses
    classifier scoring instead of regression scoring"""

    def predict(self, x, copy=True):
        # Super searches parents of MyPLSClassifier for the method predict and
        # finds it in _PLS, which is also the mehthod PLSRegression inherits
        p = super().predict(X=x, copy=copy)
        return (p > 0).astype(int)

    def score(self, X, y, sample_weight=None):
        # Super searches the MRO of MyPLSClassifier for score. The first it
        # finds is ClassifierMixin.score because ClassifierMixin is the first
        # argument in class the initiation
        return super().score(X=X, y=y, sample_weight=sample_weight)


class MyColPicker(BaseEstimator):
    def __init__(self, col_list=(1, 2)):
        self.col_list = col_list

    def fit(self, x, y):
        return self

    def transform(self, x):
        assert isinstance(x, np.ndarray)
        return x[:, self.col_list]
