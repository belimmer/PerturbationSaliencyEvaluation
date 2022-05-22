"""
This module plots the sanity check results.

Some functions are adapted from https://github.com/HuTobias/HIGHLIGHTS-LRP
Date: 2020
commit: 834bf795ee37a74b611beb79851438e9a8afd676
License: MIT
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import applications.atari.used_parameters as used_parameters


def show_and_save_plt(ax ,file_name, y_label=None, ylim =None, label_size = 18, tick_size = 14, only_plot= False):
    """
    Shows and saves the given plot and defines the appearance of the final plot.
    :param ax: the plot to be saved.
    :param file_name: save file name where the file is saved.
    :param y_label: the y axis label displayed
    :param title: titel of displayed in the plot (currently not used)
    :param ylim: limits of the y axis.
    :param label_size: font size of the label text
    :param tick_size: font size of the tick numbers
    """
    #this only works the second time the function is used, since it sets the style for future plots.
    # It was still more convenient this way. #TODO fix this
    sns.set_style("whitegrid")

    if y_label != None:
        plt.ylabel(y_label)
    plt.xlabel(None)
    if ylim != None:
        ax.set(ylim=ylim)

    try:
        ax.yaxis.label.set_size(label_size)
        ax.xaxis.label.set_size(label_size)
    except:
        try:
            plt.ylabel(y_label, fontsize=label_size)
            plt.xlabel(fontsize=label_size)
        except Exception as e:
            print(e)

    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    if only_plot:
        ax.set(xticklabels=[])
        ax.set(xlabel=None)
        ax.tick_params(bottom=False)
        ax.set(yticklabels=[])
        ax.set(ylabel=None)

    file_name = os.path.join('figures', file_name)
    if not (os.path.exists(file_name)):
        os.makedirs(file_name)
        os.rmdir(file_name)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)

    plt.show()


def normalise_image(image):
    '''normalises image by forcing the min and max values to 0 and 1 respectively
     :param image: the input image
    :return: normalised image as numpy array
    '''
    try:
        image = np.asarray(image)
    except:
        print('Cannot convert image to array')
    image = image - image.min()
    if image.max() != 0:
        image = image / image.max()
    return image


def combine_game(file_name, games):
    """
    combines the data from the different games into one data frame.
    :param file_name: the name of the result file in each game directory
    :param games: the directory names of the different games
    :return: the combined data frame
    """
    data_frame = pd.DataFrame()
    for game in games:
        dir_name = os.path.join("results", game)
        game_file_name = os.path.join(dir_name, file_name)
        game_frame = pd.read_csv(game_file_name)
        if data_frame.empty:
            data_frame = game_frame
        else:
            data_frame = data_frame.append(game_frame)

    return data_frame


def add_approach(file_name, approach_name, games, df= pd.DataFrame()):
    """
    Adds the results of a given saliency map approach to the given DataFrame.
    :param file_name: the name of the result file in each game directory
    :param approach_name: the saliency map generation approach
    :param games: the directory names of the different games
    :param df: the dataFrame which should be extended. If it its empty then a new one will be generated
    :return: the resulting dataFrame
    """
    temp_frame = combine_game(file_name, games)
    #drop the results from random saleincy maps
    temp_frame = temp_frame.drop(temp_frame[temp_frame.rand_layer > 5].index)
    temp_frame["approach"] = approach_name
    if df.empty:
        return temp_frame
    else:
        return df.append(temp_frame)


def plot_sanity_checks(data_frame, directory, only_plot = False):
    """
    Plots the sanity check results within a data_frame
    :param data_frame: the data frame with the results
    :param directory: the directory to save the plots
    :param only_plot: should only the plot be saved or also the legend etc.
    :return: nothing, the plots are saved
    """
    # remove the random saliency values
    data_frame = data_frame.drop(data_frame[data_frame.rand_layer > 5].index)

    data_frame['rand_layer'] = data_frame.rand_layer.apply(
        lambda x: 'fc_2' if x == 1 else 'fc_1' if x == 2 else 'conv_3' if x == 3 else 'conv_2' if x == 4 else
        'conv_1' if x ==5 else "uniform" if x == 6 else "gaussian" if x == 7 else "what?")

    sns.set(palette='colorblind', style="whitegrid")

    ax = sns.barplot(x='rand_layer', y='pearson', data=data_frame)
    show_and_save_plt(ax, os.path.join(directory,'pearson'), label_size=28, tick_size=40, y_label='Pearson', ylim=(0, 1), only_plot=only_plot)
    ax = sns.barplot(x='rand_layer', y='ssim', data=data_frame)
    plt.xlabel(None)
    show_and_save_plt(ax, os.path.join(directory,'ssim'), label_size=28, tick_size=40, y_label='Ssim', ylim=(0, 1), only_plot=only_plot)
    ax = sns.barplot(x='rand_layer', y='spearman', data=data_frame)
    plt.xlabel(None)
    show_and_save_plt(ax, os.path.join(directory,'spearman'), label_size=28, tick_size=40, y_label='Spearman', ylim=(0, 1), only_plot=only_plot)


def plot_combined_results(file_name, games, directory):
    """
    plots the sanity check results for a single approach
    :param file_name: the name of the saved results for the approach
    :param games: the games that should be combined
    :param directory: the directory where the plots should be saved
    :return: nothing, the plots are saved
    """
    data_frame = combine_game(file_name, games)
    plot_sanity_checks(data_frame, directory, only_plot=True)


if __name__ == '__main__':

    games = ["pacman", "breakout", "spaceInvaders", "frostbite"]

    file_name = used_parameters.OCCL_NAME + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    combined_df = add_approach(file_name, approach_name="Occlusion Sensitivity", games=games)

    BLUR = True
    RAW_DIFF = True
    file_name = used_parameters.NOISE_NAME + "_" + str(BLUR) + '_' + str(RAW_DIFF) + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    combined_df = add_approach(file_name, approach_name="NS Chosen Action", games=games, df=combined_df)

    BLUR = True
    RAW_DIFF = False
    file_name = used_parameters.NOISE_NAME + "_" + str(BLUR) + '_' + str(RAW_DIFF) + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    combined_df = add_approach(file_name, approach_name="NS Original", games=games, df=combined_df)

    BLUR = False
    RAW_DIFF = False
    file_name = used_parameters.NOISE_NAME + "_" + str(BLUR) + '_' + str(RAW_DIFF) + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    combined_df = add_approach(file_name, approach_name="NS Black", games=games, df=combined_df)

    file_name = used_parameters.RISE_NAME + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    combined_df = add_approach(file_name, approach_name="RISE", games=games, df=combined_df)

    file_name = used_parameters.SARFA_NAME + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    combined_df = add_approach(file_name, approach_name="SARFA", games=games, df=combined_df)

    file_name = used_parameters.SLIC_NAME + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    combined_df = add_approach(file_name, approach_name="LIME SLIC", games=games, df=combined_df)

    file_name = used_parameters.QUICKSHIFT_NAME + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    combined_df = add_approach(file_name, approach_name="LIME Quick", games=games, df=combined_df)

    file_name = used_parameters.FELZ_NAME + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    combined_df = add_approach(file_name, approach_name="LIME Felz", games=games, df=combined_df)

    directory = "combined"
    markers =  ('o', 'v', '^', '>', 'X', 'P', 's', 'p', 'D') #, 'h', 'H', 'D', 'd', 'P', 'X')
    plot_params = {"ci": 99, "err_style": "band", "markers": markers, "markersize" : 10, "legend":False, "dashes" : False}
    ax = sns.lineplot(x='rand_layer', y='pearson', hue="approach", style="approach", data=combined_df, **plot_params)
    show_and_save_plt(ax, os.path.join(directory, 'pearson'), label_size=28, tick_size=40, y_label='Pearson',
                     ylim=(0, 1), only_plot=True)

    ax = sns.lineplot(x='rand_layer', y='spearman', hue="approach", style="approach", data=combined_df, **plot_params)
    show_and_save_plt(ax, os.path.join(directory, 'spearman'), label_size=28, tick_size=40, y_label='Spearman',
                      ylim=(0, 1), only_plot=True)

    ax = sns.lineplot(x='rand_layer', y='ssim', hue="approach", style="approach", data=combined_df, **plot_params)
    show_and_save_plt(ax, os.path.join(directory, 'ssim'), label_size=28, tick_size=40, y_label='SSIM',
                      ylim=(0, 1), only_plot=True)

    # draw the legend
    plot_params["legend"] = "full"
    ax = sns.lineplot(x='rand_layer', y='ssim', hue="approach", style="approach", data=combined_df, **plot_params)

    handles = ax.get_legend_handles_labels()
    handles[0].pop(0)
    handles[1].pop(0)
    fig = plt.figure(figsize=(7.8, 0.6))
    fig.legend(handles[0],handles[1], loc="upper left", frameon=True, ncol= 5)
    plt.savefig(fname=os.path.join("figures","sanity_legend.png"), dpi=300)
    plt.show()


    ## LIME results
    # file_name = "Lime_slic_80_10_05_1000.csv"
    # plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    # combined_df = add_approach(file_name, approach_name="LIME SLIC", games=games)
    # file_name = "Lime_quickshift_1_4_0_3000"
    # plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    # combined_df = add_approach(file_name, approach_name="LIME Quickshift", games=games, df=combined_df)
    #
    # file_name = "Lime_felzenswalb_1_025_2_2500"
    # plot_combined_results(file_name, games, file_name.replace(".csv", ""))
    # combined_df = add_approach(file_name, approach_name="LIME Felzenszwalb", games=games, df=combined_df)
    #
    # directory = "combined_LIME"
    # plot_params = {"ci": 99, "err_style": "band", "markers": True, "markersize": 10, "legend": False}
    # ax = sns.lineplot(x='rand_layer', y='pearson', hue="approach", style="approach", data=combined_df, **plot_params)
    # show_and_save_plt(ax, os.path.join(directory, 'pearson'), label_size=28, tick_size=40, y_label='Pearson',
    #                   ylim=(0, 1), only_plot=True)
    #
    # ax = sns.lineplot(x='rand_layer', y='spearman', hue="approach", style="approach", data=combined_df, **plot_params)
    # show_and_save_plt(ax, os.path.join(directory, 'spearman'), label_size=28, tick_size=40, y_label='Spearman',
    #                   ylim=(0, 1), only_plot=True)
    #
    # ax = sns.lineplot(x='rand_layer', y='ssim', hue="approach", style="approach", data=combined_df, **plot_params)
    # show_and_save_plt(ax, os.path.join(directory, 'ssim'), label_size=28, tick_size=40, y_label='SSIM',
    #                   ylim=(0, 1), only_plot=True)
    #
    # # draw the legend
    # plot_params["legend"] = "full"
    # ax = sns.lineplot(x='rand_layer', y='ssim', hue="approach", style="approach", data=combined_df, **plot_params)
    #
    # handles = ax.get_legend_handles_labels()
    # handles[0].pop(0)
    # handles[1].pop(0)
    # fig = plt.figure(figsize=(5.3, 0.4))
    # fig.legend(handles[0], handles[1], loc="upper left", frameon=True, ncol=len(handles[0]))
    # plt.savefig(fname=os.path.join("figures", "lime_sanity_legend.png"))
    # plt.show()
