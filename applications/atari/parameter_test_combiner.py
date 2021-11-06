""" Module for combining the results from different insertion metrics."""

import numpy as np
import os
import applications.atari.evaluation_utils as evaluation_utils

import pandas as pd


def combine_aucs(segmentation_):
    """combines the aucs obtained with black and random occlusion insertion metric by taking simple mean"""
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


def load_values(filename):
    """ loads the raw parameter test results"""
    saved_values = np.load(filename, allow_pickle=True)
    parameters_ = saved_values[0]
    times_ = saved_values[1]
    q_vals_ = saved_values[2]
    return parameters_, times_, q_vals_


def calculate_aucs(load_path, save_path):
    """ calculates the AUC from the raw insertion metric results.
     Saves the results of the parameter test in a readable csv"""
    parameters, times, q_vals = load_values(load_path)

    aucs = []
    for idx in range(len(parameters)):
        auc_sum = 0
        for state_array in q_vals:
            state_q_vals = state_array[idx]
            insertion_result_ = evaluation_utils.process_single_insertion_result(state_q_vals)
            auc_sum += evaluation_utils.auc(insertion_result_)
        aucs.append(auc_sum)

    data_frame = pd.DataFrame()
    data_frame["aucs"] = aucs
    data_frame["params"] = parameters
    data_frame["time"] = times

    data_frame.to_csv(save_path)


if __name__ == '__main__':
    segmentations = ["occl", "noise","rise", "felzenswalb", "quickshift", "slic"]

    for seg in segmentations:

        # calculate the values for black occlusion insertion metric
        out_path = os.path.join("parameter_results", seg, "best_parameters_black.csv")
        res_path = os.path.join("parameter_results", seg, "best_parameters_black.npy")
        calculate_aucs(res_path, out_path)

        # calculate the values for random occlusion insertion metric
        out_path = os.path.join("parameter_results", seg, "best_parameters_random.csv")
        res_path = os.path.join("parameter_results", seg, "best_parameters_random.npy")
        calculate_aucs(res_path, out_path)

        # combine the values for random and black occlusion
        combine_aucs(seg)













