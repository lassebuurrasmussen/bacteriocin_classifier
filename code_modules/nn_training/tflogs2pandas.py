import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator


def sum_log(path: str):
    """Extract values from path to log file"""
    default_size_guidance = {
        'compressedHistograms': 1,
        'images': 1,
        'scalars': 0,  # 0 means load all
        'histograms': 1
    }
    runlog = pd.DataFrame({"metric": [], "value": [], "step": []})

    event_acc = EventAccumulator(path, default_size_guidance)
    event_acc.Reload()
    tags = event_acc.Tags()["scalars"]
    for tag in tags:
        event_list = event_acc.Scalars(tag)
        values = list(map(lambda x: x.value, event_list))
        step = list(map(lambda x: x.step, event_list))
        r = {"metric": [tag] * len(step), "value": values, "step": step}
        r = pd.DataFrame(r)
        runlog = pd.concat([runlog, r])

    # Extract additional tags from path
    description, run_type = path.split('/')[-3:-1]
    embedding = description.split('_')
    embedding, architecture, fold = embedding[0], embedding[1], embedding[-1]

    runlog['embedding'] = embedding
    runlog['architecture'] = architecture
    runlog['fold'] = fold
    runlog['run_type'] = run_type

    return runlog


def dump_to_csv(log_dir, output_dir):
    # Get all event runs from logging_dir subdirectories
    event_paths = glob.glob(os.path.join(log_dir, "*", "*", "*"))

    # Call & append
    all_log = pd.DataFrame()
    for path in event_paths:
        log = sum_log(path)
        if log is not None:
            if all_log.shape[0] == 0:
                all_log = log
            else:
                all_log = all_log.append(log, ignore_index=True)

    print(all_log.shape)
    all_log.head()

    os.makedirs(output_dir, exist_ok=True)

    print("saving to csv file")
    out_file = os.path.join(output_dir, "all_training_logs_in_one_file.csv")
    print(out_file)
    all_log.to_csv(out_file, index=None)


def extract_results_from_csv(pd_df_path, out_file_name, save_new_plot=False):
    # Read csv generated from log files
    df = pd.read_csv(f"{pd_df_path}csv_logs/all_training_logs_in_one_file.csv")

    # Isolate validation accuracy
    df_val_acc = df.query("metric == 'epoch_acc' & run_type == 'validation'")

    # Calculate the mean accuracy per step over all folds
    fold_mean: pd.DataFrame = (df_val_acc.groupby(['embedding', 'architecture',
                                                   'step'])['value'].mean()
                               .reset_index())

    # Pick the epochs with the best mean over all folds
    best_epoch = (fold_mean.set_index('step')
                  .groupby(['embedding', 'architecture'])['value'].idxmax()
                  .rename('best_epoch').reset_index())

    def best_epochs_plot():
        plt.close('all')
        plot_df = (fold_mean.groupby(['embedding', 'architecture',
                                      'step'])['value']
                   .nth(list(range(200))).reset_index())
        plot_df = plot_df.merge(best_epoch, 'left',
                                ['embedding', 'architecture'])
        facet_grid = sns.FacetGrid(data=plot_df, col='architecture',
                                   row='embedding',
                                   margin_titles=True)
        facet_grid.map(plt.plot, 'step', 'value')
        facet_grid.map(plt.vlines, 'best_epoch', ymin=.5, ymax=1, color='red')
        [ax.grid() for ax in facet_grid.axes.flatten()]

        plt.savefig("code_modules/nn_training/BAC_UNI_len2006/best_epochs.png")

    if save_new_plot:
        # Plot the chosen epoch for each setup
        best_epochs_plot()

    # Add best epochs to df
    df = pd.merge(df, best_epoch, 'left', on=['embedding', 'architecture'])

    # Remove all other epochs from data frame
    df = df[df['step'] == df['best_epoch']]

    # Make aggregated stats for each run
    stats_df = (df.groupby(['run_type', 'embedding', 'architecture',
                            'metric'])['value']
                .agg(['mean', 'std', 'size']).round(4))

    # Add best epochs to stats df
    stats_df = pd.merge(stats_df, best_epoch, 'left',
                        on=['embedding', 'architecture'])

    # Save CSV file of stats
    stats_df.to_csv(pd_df_path + out_file_name)
