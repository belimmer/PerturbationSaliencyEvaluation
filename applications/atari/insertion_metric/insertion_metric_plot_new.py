'''
Module for plotting the insertion metric results
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

from applications.atari.sanity_checks.sanity_checks_plot import show_and_save_plt


# Normalizing formula from https://arxiv.org/pdf/2001.00396.pdf
# did not use this since we wanted to show the confidence range
def normalize(arr):
    out = []
    b = arr[0]
    t_1 = arr[-1]
    for val in arr:
        out.append((val - b)/(t_1 - b))
    out = np.asarray(out)
    return out

def auc(arr):
    """
    simple Area Under the curve calculation using sum_of_values/number_of_values
    :param data:
    :return:
    """
    auc = arr.sum() / (arr.shape[0])
    print(round(auc,3))
    return auc

def load_scores(dir_name):
    # load the different data checkpoints
    x_0 = np.load(file="insertion_metric/results/" + dir_name + "/pred_100.npy", allow_pickle=True)
    x_1 = np.load(file="insertion_metric/results/" + dir_name + "/pred_200.npy", allow_pickle=True)
    x_2 = np.load(file="insertion_metric/results/" + dir_name + "/pred_300.npy", allow_pickle=True)
    x_3 = np.load(file="insertion_metric/results/" + dir_name + "/pred_400.npy", allow_pickle=True)
    x_4 = np.load(file="insertion_metric/results/" + dir_name + "/pred_500.npy", allow_pickle=True)
    x_5 = np.load(file="insertion_metric/results/" + dir_name + "/pred_600.npy", allow_pickle=True)
    x_6 = np.load(file="insertion_metric/results/" + dir_name + "/pred_700.npy", allow_pickle=True)
    x_7 = np.load(file="insertion_metric/results/" + dir_name + "/pred_800.npy", allow_pickle=True)
    x_8 = np.load(file="insertion_metric/results/" + dir_name + "/pred_900.npy", allow_pickle=True)
    x_9 = np.load(file="insertion_metric/results/" + dir_name + "/pred_1000.npy", allow_pickle=True)

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
    # average the values for each step of the insertion game over all tested inputs
    number = x.shape[0]
    x_temp = np.sum(x, axis=0)
    x_temp = x_temp / number

    print(dir_name)
    # AUC for raw softmax values
    auc(x_temp)

    return x


if __name__ == '__main__':
    GAMES = ["pacman", "breakout", "frostbite", "spaceInvaders"]
    #GAMES = ["pacman"]

    NOISE_LIME = False

    for game in GAMES:
        if game == "pacman":
            approaches = [
                "occl_4_0",
                # "occl_4_gray",

                # "noise_4_blur",
                "noise_4_black",
                # "noise_4_blur_rawDiff",

                "rise_08_18_3000",

                "Lime_quickshift_1_4_0_3000",
                # "Lime_slic_80_10_05_1000",
                # "Lime_felzenswalb_1_025_2_2500"
            ]
        elif game == "breakout":
            approaches = [
                "occl_4_0",
                # "occl_4_gray",

                # "noise_4_blur",
                # "noise_4_black",
                "noise_4_blur_rawDiff",

                "rise_08_18_3000",

                # "Lime_quickshift_1_4_0_3000",
                # "Lime_slic_80_10_05_1000",
                "Lime_felzenswalb_1_025_2_2500"
            ]
        elif game == "frostbite":
            approaches = [
                "occl_4_0",
                # "occl_4_gray",

                "noise_4_blur",
                #"noise_4_black",
                #"noise_4_blur_rawDiff",

                "rise_08_18_3000",

                "Lime_quickshift_1_4_0_3000",
                # "Lime_slic_80_10_05_1000",
                # "Lime_felzenswalb_1_025_2_2500",
                          ]
        elif game == "spaceInvaders":
            approaches = [
                "occl_4_0",
                # "occl_4_gray",

                # "noise_4_blur",
                # "noise_4_black",
                "noise_4_blur_rawDiff",

                "rise_08_18_3000",

                # "Lime_quickshift_1_4_0_3000",
                # "Lime_slic_80_10_05_1000",
                "Lime_felzenswalb_1_025_2_2500"
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
            dir_name_ = os.path.join(game, approach)
            scores = load_scores(dir_name_)
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
        show_and_save_plt(ax, os.path.join(dir_name, game + "_insertion.png"), label_size=28, tick_size=25, y_label=None,
                          only_plot=False)

        plot_params["legend"] = "full"
        ax = sns.lineplot(x="index", y='scores', hue="approach", style="approach", data=data,
                          **plot_params)
        handles = ax.get_legend_handles_labels()
        handles = handles[0]
        handles.pop(0)


        labels = ["Occlusion Sensitivity", "Noise Sensitivity", "RISE", "LIME"]
        fig = plt.figure(figsize=(6.2, 0.4))
        if NOISE_LIME:
            labels = ["NS Original", "NS Black", "NS Chosen Action", "LIME Quickshift", "LIME SLIC", "LIME Felzenszwalb"]
            fig = plt.figure(figsize=(10, 0.4))
        fig.legend(handles, labels, loc="upper left", frameon=True, ncol= len(handles))
        plt.savefig(fname=os.path.join("figures", dir_name, "insertion_legend.png"))
        plt.show()








