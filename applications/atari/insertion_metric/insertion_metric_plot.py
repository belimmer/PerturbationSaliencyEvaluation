'''
Module for plotting and evaluating the insertion metric results
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

import applications.atari.evaluation_utils as evaluation_utils
from applications.atari.sanity_checks.sanity_checks_plot import show_and_save_plt
import applications.atari.used_parameters as used_params


def load_scores(dir_name, use_advantage=True):
    """
    load the saved results and print the mean AUC.
    :param dir_name: the directoy where the results are saved
    :return: the combined results in one array
    """
    # load the different data checkpoints
    x_0 = np.load(file="results/" + dir_name + "/pred_100.npy", allow_pickle=True)
    x_1 = np.load(file="results/" + dir_name + "/pred_200.npy", allow_pickle=True)
    x_2 = np.load(file="results/" + dir_name + "/pred_300.npy", allow_pickle=True)
    x_3 = np.load(file="results/" + dir_name + "/pred_400.npy", allow_pickle=True)
    x_4 = np.load(file="results/" + dir_name + "/pred_500.npy", allow_pickle=True)
    x_5 = np.load(file="results/" + dir_name + "/pred_600.npy", allow_pickle=True)
    x_6 = np.load(file="results/" + dir_name + "/pred_700.npy", allow_pickle=True)
    x_7 = np.load(file="results/" + dir_name + "/pred_800.npy", allow_pickle=True)
    x_8 = np.load(file="results/" + dir_name + "/pred_900.npy", allow_pickle=True)
    x_9 = np.load(file="results/" + dir_name + "/pred_1000.npy", allow_pickle=True)

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

    new_x = np.zeros((x.shape[0], x.shape[1]))
    # process the q-vals
    for i in range(len(x)):
        new_x[i] = evaluation_utils.process_single_insertion_result(x[i], use_advantage=use_advantage)

    x = new_x

    return x


if __name__ == '__main__':
    GAMES = ["pacman", "breakout", "frostbite", "spaceInvaders"]

    approaches = [
        used_params.OCCL_NAME,
        used_params.NOISE_NAME + "_1",
        used_params.SARFA_NAME,
        used_params.RISE_NAME,
        used_params.FELZ_NAME,
        used_params.QUICKSHIFT_NAME,
        used_params.SLIC_NAME,
        "rnd_baseline"
    ]

    auc_dataframe = pd.DataFrame()
    auc_dataframe["Approach"] = approaches

    for game in GAMES:
        for INSERTION_COLOR in ["random_insertion", "black_insertion"]:
            for ADVANTAGE in range(0,2):
                print(game)

                # for mean auc per state
                mean_aucs = []
                stds = []

                first = True
                for approach in approaches:
                    if INSERTION_COLOR == "combined":
                        dir_name_ = os.path.join(game, "random_insertion", approach)
                        scores = load_scores(dir_name_, use_advantage=ADVANTAGE)
                        dir_name_ = os.path.join(game, "black_insertion", approach)
                        scores2 = load_scores(dir_name_, use_advantage=ADVANTAGE)
                        scores = np.concatenate((scores,scores2), axis=0)
                    else:
                        dir_name_ = os.path.join(game, INSERTION_COLOR, approach)
                        scores = load_scores(dir_name_, use_advantage=ADVANTAGE)
                    # for mean auc per state
                    mean_auc, std = evaluation_utils.calculate_aucs(scores)
                    mean_aucs.append(mean_auc)
                    stds.append(std)
                    # for plotting the insertion result over all states
                    temp_data = pd.DataFrame()
                    temp_scores = []
                    temp_indizes = []
                    for run in scores:
                        for i in range(len(run)):
                            temp_scores.append(run[i])
                            temp_indizes.append(i)
                    temp_data["scores"] = temp_scores
                    temp_data["index"] = temp_indizes
                    temp_data["approach"] = approach
                    temp_data = temp_data.reset_index()
                    if first:
                        data = temp_data
                        first = False
                    else:
                        data = data.append(temp_data)

                # for mean auc per state
                auc_dataframe["mean_auc_" + game + "_" + str(ADVANTAGE) + "_"+ INSERTION_COLOR] = mean_aucs
                auc_dataframe["std_" + game + "_" + str(ADVANTAGE) + "_" + INSERTION_COLOR] = stds

                # plotting the insertion result over all states
                sns.set(palette='colorblind', style="whitegrid")
                plot_params = {"ci": 99, "err_style": "band", "markersize": 10, "legend": False, "dashes" : False}
                ax = sns.lineplot(x="index", y='scores', hue="approach", style="approach", data=data,
                                  **plot_params)
                ax.set(xticklabels=[])
                ax.set(xlabel=None)
                ax.set(ylabel=None)

                if ADVANTAGE:
                    dir_name = "advantage_" + INSERTION_COLOR
                else:
                    dir_name = "qVals_" + INSERTION_COLOR
                show_and_save_plt(ax, os.path.join(dir_name, game + "_insertion.png"), label_size=28, tick_size=25, y_label=None,
                                  only_plot=False)

                plot_params["legend"] = "full"
                ax = sns.lineplot(x="index", y='scores', hue="approach", style="approach", data=data,
                                  **plot_params)
                handles = ax.get_legend_handles_labels()
                handles = handles[0]
                handles.pop(0)

                labels = approaches
                fig = plt.figure(figsize=(20.2, 0.4))

                fig.legend(handles, labels, loc="upper left", frameon=True, ncol= len(handles))
                plt.savefig(fname=os.path.join("figures", dir_name, "insertion_legend.png"))
                plt.show()

        # save the mean auc over states
        auc_dataframe.to_csv(os.path.join("results",game + "_mean_AUCS.csv"))
        auc_dataframe = pd.DataFrame()
        auc_dataframe["Approach"] = approaches

