'''
This module was adapted from a module in https://github.com/HuTobias/HIGHLIGHTS-LRP
Date: 2020
commit: 834bf795ee37a74b611beb79851438e9a8afd676
License: MIT


This module implements sanity checks for saliency maps.
To this end the layers in the model are cascadingly randomized and for each step we create a copy of the model.
Then we create gameplay and saliency map streams for each of those models, using the decisions of the original model,
 such that all models get the same input states.
Finally we compare the generated saliency of all models.
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
import pandas as pd
import keras
import seaborn as sns
import gym
import skimage.segmentation as seg

from applications.atari.custom_atari_wrapper import atari_wrapper
from applications.atari.explanation import explainer

import tensorflow as tf
import keras.backend as K

import timeit

def show_and_save_plt(ax ,file_name, y_label=None, ylim =None, label_size = 18, tick_size = 14):
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

    file_name = os.path.join('figures/sanity_checks', file_name)
    if not (os.path.isdir(file_name)):
        os.makedirs(file_name)
        os.rmdir(file_name)
    plt.tight_layout()
    plt.savefig(file_name)

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

def plot_sanity_checks(data_frame, directory):
    # Create plots
    data_frame['rand_layer'] = data_frame.rand_layer.apply(
        lambda x: 'fc_2' if x == 1 else 'fc_1' if x == 2 else 'conv_3' if x == 3 else 'conv_2' if x == 4 else
        'conv_1' if x ==5 else "uniform" if x == 6 else "gaussian" if x == 7 else "what?")

    sns.set(palette='colorblind', style="whitegrid")

    ax = sns.barplot(x='rand_layer', y='pearson', data=data_frame)
    show_and_save_plt(ax, os.path.join(directory,'pearson'), label_size=28, tick_size=20, y_label='Pearson', ylim=(0, 1))
    ax = sns.barplot(x='rand_layer', y='ssim', data=data_frame)
    plt.xlabel(None)
    show_and_save_plt(ax, os.path.join(directory,'ssim'), label_size=28, tick_size=20, y_label='Ssim', ylim=(0, 1))
    ax = sns.barplot(x='rand_layer', y='spearman', data=data_frame)
    plt.xlabel(None)
    show_and_save_plt(ax, os.path.join(directory,'spearman'), label_size=28, tick_size=20, y_label='Spearman', ylim=(0, 1))


def plot_combined_results(file_name, games, directory):
    data_frame = combine_game(file_name, games)
    plot_sanity_checks(data_frame, directory)


def plot_sanity_check_results(file_name):
    data_frame = pd.read_csv(file_name)

    plot_sanity_checks(data_frame)


if __name__ == '__main__':

    games = ["pacman", "breakout", "spaceInvaders", "frostbite"]
    # games = ["pacman"]

    APPROACH = "occl"
    PATCH_SIZE = 4
    COLOR = 0
    file_name = APPROACH + '_' + str(PATCH_SIZE) + '_' + str(COLOR) + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))

    APPROACH = "noise"
    RADIUS = 4
    BLUR = False
    RAW_DIFF = False
    file_name = APPROACH + '_' + str(BLUR) + '_' + str(RAW_DIFF) + '_' + str(RADIUS) + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))

    BLUR = True
    file_name = APPROACH + '_' + str(BLUR) + '_' + str(RAW_DIFF) + '_' + str(RADIUS) + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))

    RAW_DIFF = True
    file_name = APPROACH + '_' + str(BLUR) + '_' + str(RAW_DIFF) + '_' + str(RADIUS) + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))

    file_name = "Lime_slic_80_100_0.csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))

    file_name = "Lime_quickshift_1_7_015"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))

    file_name = "Lime_felzenswalb_71_4e-1_0"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))

    APPROACH = "rise"
    MASK_SIZE = 18
    NUM_MASKS = 3000
    file_name = APPROACH + '_' + "08" + '_' + str(MASK_SIZE) + '_' + str(NUM_MASKS) + ".csv"
    plot_combined_results(file_name, games, file_name.replace(".csv", ""))



    # ####Plotting
    # for GAME in games:
    #     dir_name = os.path.join("results", GAME)
    #     APPROACH = "occl"
    #     PATCH_SIZE = 4
    #     COLOR = 0
    #     file_name = APPROACH + '_' + str(PATCH_SIZE) + '_' + str(COLOR) + ".csv"
    #     file_name = os.path.join(dir_name, file_name)
    #     plot_sanity_check_results(file_name)

    #
    #
    #     APPROACH = "noise"
    #     RADIUS = 4
    #     BLUR = False
    #     RAW_DIFF = False
    #     # file_name = APPROACH + '_' + str(BLUR) + '_' + str(RAW_DIFF) + '_' + str(RADIUS) + ".csv"
    #     # file_name = os.path.join(dir_name, file_name)
    #     # plot_sanity_check_results(file_name)
    #
    #     BLUR = True
    #     # file_name = APPROACH + '_' + str(BLUR) + '_' + str(RAW_DIFF) + '_' + str(RADIUS) + ".csv"
    #     # file_name = os.path.join(dir_name, file_name)
    #     # plot_sanity_check_results(file_name)
    #
    #     RAW_DIFF = True
    #     file_name = APPROACH + '_' + str(BLUR) + '_' + str(RAW_DIFF) + '_' + str(RADIUS) + ".csv"
    #     file_name = os.path.join(dir_name, file_name)
    #     plot_sanity_check_results(file_name)

        # file_name = "Lime_slic_80_100_0.csv"
        # file_name = os.path.join(dir_name, file_name)
        # plot_sanity_check_results(file_name)
        #
        # file_name = "Lime_quickshift_1_7_015"
        # file_name = os.path.join(dir_name, file_name)
        # plot_sanity_check_results(file_name)
        #
        # file_name = "Lime_felzenswalb_71_4e-1_0"
        # file_name = os.path.join(dir_name, file_name)
        # plot_sanity_check_results(file_name)

        # APPROACH = "rise"
        # MASK_SIZE = 18
        # NUM_MASKS = 3000
        # file_name = APPROACH + '_' + "08" + '_' + str(MASK_SIZE) +  '_' + str(NUM_MASKS) + ".csv"
        # file_name = os.path.join(dir_name, file_name)
        # plot_sanity_check_results(file_name)