"""
Module for calculating the mean run-time per saliency map approach
"""

import numpy as np
import os


def load_scores(dir_name):
    """
    load the saved run times.
    :param dir_name: the directoy where the times are saved
    :return: the combined results in one array
    """
    # load the different data checkpoints
    x_0 = np.load(file="results/" + dir_name + "/times_100.npy", allow_pickle=True)
    x_1 = np.load(file="results/" + dir_name + "/times_200.npy", allow_pickle=True)
    x_2 = np.load(file="results/" + dir_name + "/times_300.npy", allow_pickle=True)
    x_3 = np.load(file="results/" + dir_name + "/times_400.npy", allow_pickle=True)
    x_4 = np.load(file="results/" + dir_name + "/times_500.npy", allow_pickle=True)
    x_5 = np.load(file="results/" + dir_name + "/times_600.npy", allow_pickle=True)
    x_6 = np.load(file="results/" + dir_name + "/times_700.npy", allow_pickle=True)
    x_7 = np.load(file="results/" + dir_name + "/times_800.npy", allow_pickle=True)
    x_8 = np.load(file="results/" + dir_name + "/times_900.npy", allow_pickle=True)
    x_9 = np.load(file="results/" + dir_name + "/times_1000.npy", allow_pickle=True)

    # combine the data checkpoints to one array
    x = np.concatenate((x_0, x_1), axis=0)
    x = np.concatenate((x, x_2), axis=0)
    x = np.concatenate((x, x_3), axis=0)
    x = np.concatenate((x, x_4), axis=0)
    x = np.concatenate((x, x_5), axis=0)
    x = np.concatenate((x, x_6), axis=0)
    x = np.concatenate((x, x_7), axis=0)
    x = np.concatenate((x, x_8), axis=0)
    x = np.concatenate((x, x_9), axis=0)

    return x

approaches = [
    # comparison values:
    "Lime_quickshift_1_4_0_3000",
    "Lime_slic_80_10_05_1000",
    "Lime_felzenswalb_1_025_2_2500",

    "occl_4_0",
    "occl_4_gray",

    "rise_08_18_3000",

    "noise_4_blur",
    "noise_4_black",
    "noise_4_blur_rawDiff"
              ]

games = ["pacman", "breakout", "spaceInvaders", "frostbite"]
for approach in approaches:
    first = True
    for game in games:
        dir_name_ = os.path.join(game, approach)
        scores = load_scores(dir_name_)
        if first:
            data = scores
            first = False
        else:
            data = np.concatenate((data,scores))
    print(approach)
    print(data.mean())







