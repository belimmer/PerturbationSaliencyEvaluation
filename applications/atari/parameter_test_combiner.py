""" Module for combining the results from different insertion metrics."""

import numpy as np
import os

import pandas as pd

def combine_aucs(segmentation_):
    results_path = os.path.join("parameter_results", segmentation_)

    black_values = pd.read_csv(os.path.join(results_path, "best_parameters_black.csv"))
    random_values = pd.read_csv(os.path.join(results_path, "best_parameters_random.csv"))

    black_aucs = black_values["aucs"].values
    random_aucs = random_values["aucs"].values

    black_aucs = np.asarray(black_aucs)
    random_aucs = np.asarray(random_aucs)

    combined_aucs = black_aucs + random_aucs
    combined_aucs = combined_aucs / 2

    black_values["aucs"] = combined_aucs

    black_values.to_csv(os.path.join(results_path, "best_parameters_mean.csv"))


if __name__ == '__main__':
    segmentations = ["occl", "noise","rise", "felzenswalb", "quickshift", "slic"]

    for seg in segmentations:
        combine_aucs(seg)



