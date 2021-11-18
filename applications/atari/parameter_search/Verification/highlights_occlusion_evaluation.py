""" Module for evaluating the parameter search results for various 10 staste subsets of the game Pacman
 and comparing them with the results of a parameter search on 1000 states of the game Pacman.
This can then be used to verify which states are suited to search for good parameters."""

import os
import pandas as pd
from applications.atari.parameter_search.parameter_test_combiner import load_values, calculate_aucs, load_combined_values
from scipy import stats


def evaluate_parameter_search(results_dir, ground_truth_path, use_advantage):
    """evaluates the parameter search in the *results_dir* and compares it to the auc values in *ground_truth_path*"""
    # calculate the values for black occlusion insertion metric
    out_path = os.path.join(results_dir, "best_parameters_black.csv")
    black_path = os.path.join(results_dir, "best_parameters_black.npy")
    params, time_vals, q_values = load_values(black_path)
    calculate_aucs(params, time_vals, q_values, out_path, use_advantage)

    # calculate the values for random occlusion insertion metric
    out_path = os.path.join(results_dir, "best_parameters_random.csv")
    random_path = os.path.join(results_dir, "best_parameters_random.npy")
    params, time_vals, q_values = load_values(random_path)
    calculate_aucs(params, time_vals, q_values, out_path, use_advantage)

    # combine the values for random and black occlusion
    out_path = os.path.join(results_dir, "best_parameters_mean.csv")
    params, time_vals, q_values = load_combined_values(black_path, random_path)
    calculate_aucs(params, time_vals, q_values, out_path, use_advantage)

    # load the results from the parameter search using only highlight states
    results_path = os.path.join(results_dir, "best_parameters_mean.csv")
    highlight_state_results = pd.read_csv(results_path)

    # load the results from the full parameter search on 1000 states of Pacman
    full_results = pd.read_csv(ground_truth_path)

    # calculate the correlation values
    highlight_state_aucs = highlight_state_results["aucs"].values
    full_aucs = full_results["mean auc"].values
    spearman_correlation = stats.spearmanr(highlight_state_aucs, full_aucs)
    kendalls_tau = stats.kendalltau(highlight_state_aucs, full_aucs)

    return spearman_correlation, kendalls_tau


if __name__ == '__main__':
    seg_ = "occl"
    USE_ADVANTAGE = True

    if USE_ADVANTAGE:
        full_results_file = "occlusion_results_advantage.csv"
        out_file = "correlation_values_advantage.csv"
    else:
        full_results_file = "occlusion_results_rawValue_divided.csv"
        out_file = "correlation_values_rawValue_divided.csv"

    states = []
    spearmans = []
    kendalls = []

    # search parameters for the random states chosen in select_random_states.py
    for i in range(10):
        state_path_ = "random_states_" + str(i)
        results_dir_ = os.path.join(state_path_, "parameter_search")
        spearman, kendall = evaluate_parameter_search(results_dir_, ground_truth_path=full_results_file,
                                                      use_advantage=USE_ADVANTAGE)
        spearmans.append(spearman)
        kendalls.append(kendall)
        states.append(state_path_)

    for config in ["thres10", "thres20", "thres30", "thres40","pos_neg_thres10", "pos_neg_thres40", "diverse_importance",
                   "thres25", "thres28", "thres32", "thres33", "thres35"]:
        state_path_ = "highlight_states_" + config + "/"
        results_dir_ = os.path.join(state_path_, "parameter_search")
        spearman, kendall = evaluate_parameter_search(results_dir_, ground_truth_path=full_results_file,
                                                      use_advantage=USE_ADVANTAGE)
        spearmans.append(spearman)
        kendalls.append(kendall)
        states.append(state_path_)

    data_frame = pd.DataFrame()
    data_frame["Used states"] = states
    data_frame["Spearman Correlation to full data"] = spearmans
    data_frame["Kendall's Tau"] = kendalls
    data_frame.to_csv(out_file)















