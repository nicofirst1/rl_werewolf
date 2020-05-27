import csv
import math
from collections import Counter

import numpy as np


def vote_difference(cur_targets, prev_targets):
    """
    Return the number of different cells in two matrices
    :param cur_targets: ndarray
    :param prev_targets: ndarray
    :return: float, normalize difference
    """

    assert cur_targets.size == prev_targets.size, "matrices size must be the same"

    num_player = len(cur_targets)
    diff = np.sum(cur_targets != prev_targets) / (num_player ** 2)

    return diff


def measure_influence(cur_targets, prev_targets, flexibility):
    def filter_dead_players():
        """
        Remove rows with -1 from both matrices
        :return:
        """
        to_remove = np.concatenate((np.argwhere(cur_targets[:, :1] == -1), np.argwhere(prev_targets[:, :1] == -1)))
        to_remove = to_remove[:, 0]
        to_remove = np.unique(to_remove)

        ct = cur_targets
        pt = prev_targets

        ct = np.delete(ct, to_remove, axis=0)
        pt = np.delete(pt, to_remove, axis=0)

        return ct, pt

    ct, pt = filter_dead_players()

    if not len(ct): return 0

    # get just hte visible part
    pt = pt[:, :flexibility]
    ct = ct[:, :flexibility]

    # count number of repetition
    wtp = Counter(pt.flatten())
    wtc = Counter(ct.flatten())

    # normalize by size
    wtp = {k: v / np.size(pt) for k, v in wtp.items()}
    wtc = {k: v / np.size(ct) for k, v in wtc.items()}

    # add up
    diff = 0
    for k in wtp.keys():
        try:
            diff += math.pow(wtp[k] - wtc[k], 2)
        except KeyError:
            diff += wtp[k]

    return 1 - diff


def load_analyze_csv(path2file, min_step=0):
    """
    Analyze the csv output of tensorboard, printing mean and std for the last min_step components
    Parameters
    ----------
    path2file : str, path to csv file
    min_step : int, last steps to take from the csv

    Returns
    -------

    """

    with open(path2file, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)[-min_step:]
        data = data[1:]
        data = [elem[2] for elem in data]
        data = [float(elem) for elem in data]
        print(f"{np.mean(data)} +- {np.std(data)}")

# load_analyze_csv("/home/dizzi/Downloads/accord_mean.csv",min_step=100)
