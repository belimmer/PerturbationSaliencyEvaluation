'''
Module for evaluating the full parameter search conducted by full_occlusion_search.py for the game Pacman with fast parameter configurations of occlusion.
This can then be used to verify which states are suited to search for good parameters.

This script needs to be run from the insertion_metric folder, since the insertion results are saved there.
'''

import numpy as np
import os
import pandas as pd

from applications.atari.evaluation_utils import calculate_aucs
from applications.atari.insertion_metric.insertion_metric_plot import load_scores


if __name__ == '__main__':
    GAMES = ["pacman"]

    # the color used perturbing the image in the insertion metric
    INSERTION_COLOR = "combined"

    for game in GAMES:
        print(game)

        means = []
        stds = []
        params = []

        for i in range(4, 11):
            for j in range(0, 2):
                for k in range(0, 2):
                    patch_size = i
                    color = 0.5 * j
                    softmax = k
                    parameters = (patch_size, color, softmax)

                    approach = "occl_" + str(patch_size) + '_' + str(color) + '_' + str(softmax)


                    if INSERTION_COLOR == "combined":
                        dir_name_ = os.path.join(game, "random_insertion", approach)
                        scores = load_scores(dir_name_, use_advantage=True)
                        dir_name_ = os.path.join(game, "black_insertion", approach)
                        scores2 = load_scores(dir_name_, use_advantage=True)
                        scores = np.concatenate((scores,scores2), axis=0)
                    else:
                        dir_name_ = os.path.join(game, INSERTION_COLOR, approach)
                        scores = load_scores(dir_name_, use_advantage=True)
                    print(approach)
                    mean, std = calculate_aucs(scores)

                    params.append(parameters)
                    means.append(mean)
                    stds.append(std)

        temp_data = pd.DataFrame()
        temp_data["parameters"] = params
        temp_data["mean auc"] = means
        temp_data["standard deviation"] = stds

        temp_data.to_csv("occlusion_results.csv")








