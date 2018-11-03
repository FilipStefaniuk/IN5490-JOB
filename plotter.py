import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import auc

sns.set(style='darkgrid')


class Experiment(object):

    def __init__(self, name, params_file, *results_files):
        """Initialize the experiment.

            Args:
                params_file: filename or file_buffer with parameters (in json format)
                results_files: list of filenames or buffers with results in csv format
        """
        try:
            params_file = open(params_file)
        except:
            pass

        self.name = name
        self.params = json.load(params_file)
        self.results = [pd.read_csv(csv_file) for csv_file in results_files]

    def get_auc(self, y='EpRewMean', x='TimestepsSoFar', min_val=0, max_val=1):
        auc_vals = [auc(df[x] / max(df[x]), (df[y] - min_val) / (max_val - min_val)) for df in self.results]
        return auc_vals


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('params_dir')
    parser.add_argument('results_dir', nargs='+')

    return parser.parse_args()


def get_experiments(params_dir, *results_dir, csv_file='progress.csv'):
    experiments = []

    for filename in os.listdir(params_dir):
        experiments.append(Experiment(
            os.path.splitext(filename)[0],
            os.path.join(params_dir, filename),
            *[os.path.join(result_dir, os.path.splitext(filename)[0],
                           csv_file) for result_dir in results_dir]
        ))

    return experiments


def plot_learning_curve(experiments, **kwargs):
    param = list(experiments[0].params.keys())[0]

    data = []
    for experiment in experiments:
        tmp = pd.concat([result[['eprewmean', 'total_timesteps']] for result in experiment.results])
        tmp[param] = experiment.params[param]
        data.append(tmp)

    data = pd.concat(data)
    sns.relplot(x='total_timesteps', y='eprewmean', data=data, kind='line', hue=param)


def plot_auc_2d(experiments, xscale='linear', **kwargs):

    param = list(experiments[0].params.keys())[0]
    data = [(*experiment.params.values(), auc)
            for experiment in experiments
            for auc in experiment.get_auc(**kwargs)]

    data = pd.DataFrame(data, columns=[param, 'ALC'])

    _, ax = plt.subplots()
    ax.set(xscale=xscale)
    sns.relplot(x=param, y='ALC', data=data, ax=ax, kind='line')
    plt.close(2)


def plot_heatmap(experiments, **kwargs):

    param1, param2 = list(experiments[0].params.keys())
    data = [(experiment.params[param1], experiment.params[param2], auc)
            for experiment in experiments
            for auc in experiment.get_auc(**kwargs)]

    data = pd.DataFrame(data, columns=[param1, param2, 'ALC'])
    data = data.pivot(param1, param2, "ALC")
    ax = sns.heatmap(data)


def main():
    args = get_args()
    experiments = get_experiments(args.params_dir, *args.results_dir)

    # plot_learning_curve(experiments[:10], min_val=-200, max_val=200)
    # plot_heatmap(experiments, min_val=-200, max_val=200)
    # plot_auc_2d(experiments, min_val=-200, max_val=200)
    # plot_auc_2d(experiments, x='total_timesteps', y='eprewmean', min_val=-200, max_val=200)
    # plt.show()

if __name__ == '__main__':
    main()
