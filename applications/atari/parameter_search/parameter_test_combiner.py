""" Module for combining the results from different insertion metrics."""

import numpy as np
import os
import applications.atari.evaluation_utils as evaluation_utils

import pandas as pd


def load_combined_values(path1, path2):
    """combines the values obtained with black and random occlusion insertion metric"""
    black_params, black_times, black_q_vals = load_values(path1)
    random_params, random_times, random_q_vals = load_values(path2)

    # both tests should have been done with the same parameters
    assert((black_params == random_params))

    combined_times = np.asarray(black_times) + np.asarray(random_times)
    combined_q_vals = black_q_vals + random_q_vals

    return black_params, combined_times, combined_q_vals

def load_values(filename):
    """ loads the raw parameter test results"""
    saved_values = np.load(filename, allow_pickle=True)
    parameters_ = list(saved_values[0])
    times_ = saved_values[1]
    q_vals_ = saved_values[2]
    return parameters_, times_, q_vals_


def calculate_aucs(parameters, times, q_vals, save_path, use_advantage=True):
    """ calculates the AUC from the raw insertion metric results.
     Saves the results of the parameter test in a readable csv"""

    aucs = []
    stds = []
    for idx in range(len(parameters)):
        auc_list = []
        for state_array in q_vals:
            state_q_vals = state_array[idx]
            insertion_result_ = evaluation_utils.process_single_insertion_result(state_q_vals,
                                                                                 use_advantage=use_advantage)
            auc_list.append(evaluation_utils.auc(insertion_result_))
        auc_list = np.asarray(auc_list)
        stds.append(auc_list.std())
        aucs.append(auc_list.mean())

    data_frame = pd.DataFrame()
    data_frame["aucs"] = aucs
    data_frame["params"] = parameters
    data_frame["Standard deviation"] = stds
    data_frame["time"] = times

    data_frame.to_csv(save_path)


if __name__ == '__main__':
    segmentations = ["occl", "noise","rise", "felzenswalb", "quickshift", "slic", "sarfa"]

    results_dir = "parameter_results"

    for seg in segmentations:
        for use_adv in range(0,2):
            if use_adv:
                prefix = "advantage"
            else:
                prefix = "qval"
            # calculate the values for black occlusion insertion metric
            out_path = os.path.join(results_dir, seg, prefix + "_parameters_black.csv")
            black_path = os.path.join(results_dir, seg, "best_parameters_black.npy")
            params, time_vals, q_values = load_values(black_path)
            calculate_aucs(params, time_vals, q_values, out_path, use_advantage=use_adv)

            # calculate the values for random occlusion insertion metric
            out_path = os.path.join(results_dir, seg, prefix + "_parameters_random.csv")
            random_path = os.path.join(results_dir, seg, "best_parameters_random.npy")
            params, time_vals, q_values = load_values(random_path)
            calculate_aucs(params, time_vals, q_values, out_path, use_advantage=use_adv)

            # combine the values for random and black occlusion
            out_path = os.path.join(results_dir, seg, prefix + "_parameters_mean.csv")
            params, time_vals, q_values = load_combined_values(black_path, random_path)
            calculate_aucs(params, time_vals, q_values, out_path, use_advantage=use_adv)


        # COMBINE the advantage and q_val results using
        advantage_results = pd.read_csv(os.path.join(results_dir, seg, "advantage" + "_parameters_mean.csv"))
        q_val_results = pd.read_csv(os.path.join(results_dir, seg, "qval" + "_parameters_mean.csv"))

        advantage_aucs = advantage_results["aucs"].values
        q_val_aucs = q_val_results["aucs"].values

        mean_advantage = np.mean(advantage_aucs)
        std_advantage = np.std(advantage_aucs)

        mean_q_vals = np.mean(q_val_aucs)
        std_q_vals = np.std(q_val_aucs)

        #calculate z values
        advantage_aucs = (advantage_aucs - mean_advantage)/std_advantage
        std_q_vals = (q_val_aucs - mean_q_vals)/ std_q_vals


        mean_aucs = advantage_aucs + std_q_vals

        final_df = pd.DataFrame()
        final_df["aucs"] = mean_aucs
        final_df["param" \
                 "s"] = advantage_results["params"]
        final_df["time"] = advantage_results["time"] / 20

        final_df.to_csv(os.path.join(results_dir, seg, "final_parameters_results.csv"))













