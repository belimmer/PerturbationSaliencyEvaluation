'''
Module for plotting the insertion metric results
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

import applications.atari.evaluation_utils as evaluation_utils
from applications.atari.sanity_checks.sanity_checks_plot import show_and_save_plt


def load_scores(dir_name):
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
        new_x[i] = evaluation_utils.process_single_insertion_result(x[i])

    x = new_x

    # print the AUC
    # print(dir_name)
    # evaluation_utils.auc(x_temp)

    return x


if __name__ == '__main__':
    GAMES = ["pacman", "breakout", "frostbite", "spaceInvaders"]

    # Do you want to generate plots for all Noise Sensitivity and LIME variants?
    NOISE_LIME = False

    # the color used perturbing the image in the insertion metric
    INSERTION_COLOR = "combined"

    for game in GAMES:
        print(game)
        if game == "pacman":
            approaches = [
                "occl_4_0",
                # "occl_4_gray",

                "rise_08_18_3000",

                "Lime_quickshift_1_4_0_3000",
                # "Lime_slic_80_10_05_1000",
                # "Lime_felzenswalb_1_025_2_2500",

                # "noise_4_blur",
                "noise_4_black",
                # "noise_4_blur_rawDiff"
            ]
        elif game == "breakout":
            approaches = [
                "occl_4_0",
                # "occl_4_gray",

                "rise_08_18_3000",

                # "Lime_quickshift_1_4_0_3000",
                # "Lime_slic_80_10_05_1000",
                "Lime_felzenswalb_1_025_2_2500",

                # "noise_4_blur",
                # "noise_4_black",
                "noise_4_blur_rawDiff"
            ]
        elif game == "frostbite":
            approaches = [
                "occl_4_0",
                # "occl_4_gray",

                "rise_08_18_3000",

                "Lime_quickshift_1_4_0_3000",
                # "Lime_slic_80_10_05_1000",
                # "Lime_felzenswalb_1_025_2_2500",

                "noise_4_blur",
                #"noise_4_black",
                #"noise_4_blur_rawDiff"
                          ]
        elif game == "spaceInvaders":
            approaches = [
                "occl_4_0",
                # "occl_4_gray",

                "rise_08_18_3000",

                # "Lime_quickshift_1_4_0_3000",
                # "Lime_slic_80_10_05_1000",
                "Lime_felzenswalb_1_025_2_2500",

                # "noise_4_blur",
                # "noise_4_black",
                "noise_4_blur_rawDiff"
            ]

        if NOISE_LIME:
            approaches = [
                "noise_4_blur",
                "noise_4_black",
                "noise_4_blur_rawDiff",
                "Lime_quickshift_1_4_0_3000",
                "Lime_slic_80_10_05_1000",
                "Lime_felzenswalb_1_025_2_2500"
            ]


        # Plot settings
        first = True
        for approach in approaches:
            if INSERTION_COLOR == "combined":
                dir_name_ = os.path.join(game, "random_insertion", approach)
                scores = load_scores(dir_name_)
                dir_name_ = os.path.join(game, "black_insertion", approach)
                scores2 = load_scores(dir_name_)
                scores = np.concatenate((scores,scores2), axis=0)
            else:
                dir_name_ = os.path.join(game, INSERTION_COLOR, approach)
                scores = load_scores(dir_name_)
            print(approach)
            mean_auc, std = evaluation_utils.calculate_aucs(scores)
            print(mean_auc)
            print(std)
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

        sns.set(palette='colorblind', style="whitegrid")
        plot_params = {"ci": 99, "err_style": "band", "markersize": 10, "legend": False}
        ax = sns.lineplot(x="index", y='scores', hue="approach", style="approach", data=data,
                          **plot_params)
        ax.set(xticklabels=[])
        ax.set(xlabel=None)
        # ax.tick_params(bottom=False)
        # ax.set(yticklabels=[])
        ax.set(ylabel=None)

        dir_name = "insertion_top"
        if NOISE_LIME:
            dir_name = "insertion_noise_and_lime"
        show_and_save_plt(ax, os.path.join(dir_name, game + "_insertion_" + INSERTION_COLOR + ".png"), label_size=28, tick_size=25, y_label=None,
                          only_plot=False)

        plot_params["legend"] = "full"
        ax = sns.lineplot(x="index", y='scores', hue="approach", style="approach", data=data,
                          **plot_params)
        handles = ax.get_legend_handles_labels()
        handles = handles[0]
        handles.pop(0)


        labels = ["Occlusion Sensitivity", "RISE", "LIME", "Noise Sensitivity"]
        fig = plt.figure(figsize=(6.2, 0.4))
        if NOISE_LIME:
            labels = ["NS Original", "NS Black", "NS Chosen Action", "LIME Quickshift", "LIME SLIC", "LIME Felzenszwalb"]
            fig = plt.figure(figsize=(10.2, 0.4))
        fig.legend(handles, labels, loc="upper left", frameon=True, ncol= len(handles))
        plt.savefig(fname=os.path.join("figures", dir_name, "insertion_legend.png"))
        plt.show()








