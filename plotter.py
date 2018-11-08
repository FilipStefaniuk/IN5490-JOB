import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import auc

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

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


def plot_learning_curve(experiment, y='EpRewMean', x='TimestepsSoFar', **kwargs):

    data = []
    for i, result in enumerate(experiment.results):
        tmp = result[[x, y]]
        tmp['run'] = i
        data.append(tmp)

    data = pd.concat(data)
    sns.relplot(x=x, y=y, data=data, kind='line', hue='run', **kwargs)


def plot_auc_2d(experiments, xscale='linear', ax=None, **kwargs):

    param = list(experiments[0].params.keys())[0]
    data = [(*experiment.params.values(), auc)
            for experiment in experiments
            for auc in experiment.get_auc(**kwargs)]

    data = pd.DataFrame(data, columns=[param, 'ALC'])

    if not ax:
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

def plot_3d(experiments, interpolate, **kwargs):
    param1, param2 = list(experiments[0].params.keys())
    data = np.array([np.array((experiment.params[param1], experiment.params[param2], auc))
            for experiment in experiments
            for auc in experiment.get_auc(**kwargs)])

    cMap = plt.cm.Spectral
    n = 40
    nj = 40j

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if interpolate:
        gridX, gridY = np.mgrid[min(data[:,0]):max(data[:,0]):nj, min(data[:,1]):max(data[:,1]):nj]
        dataInPol = griddata(data[:,0:2], data[:,2], (gridX, gridY), method='linear')
        gridX = gridX.reshape(n*n)
        gridY = gridY.reshape(n*n)
        dataInPol = dataInPol.reshape(n*n)
        ax.plot_trisurf(gridX, gridY, dataInPol, cmap=cMap, linewidth=0)
    else:
        ax.plot_trisurf(data[:,0], data[:,1], data[:,2], cmap=cMap, linewidth=0)
    plt.xlabel(param1)
    plt.ylabel(param2)

    return fig

def main():
    args = get_args()
    experiments = get_experiments(args.params_dir, *args.results_dir)


    # plot_learning_curve(experiments[:10], min_val=-200, max_val=200)
    # plot_heatmap(experiments, min_val=-200, max_val=200)
    # plot_auc_2d(experiments, min_val=-200, max_val=200)
    # plot_auc_2d(experiments, x='total_timesteps', y='eprewmean', min_val=-200, max_val=200)
    plot_3d(experiments, False, min_val=-200, max_val=200)
    plot_3d(experiments, True, min_val=-200, max_val=200)
    plt.show()

if __name__ == '__main__':
    main()
