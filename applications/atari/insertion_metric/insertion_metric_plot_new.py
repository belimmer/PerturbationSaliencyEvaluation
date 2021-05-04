'''
Module for plotting the insertion metric results
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


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
    print(auc)
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
    x = np.sum(x, axis=0)
    x = x / number

    print(dir_name)
    # AUC for raw softmax values
    auc(x)

    # # normalize prediction values to have minimum 0 and maximum 1
    # x_normalized = normalize(x)
    #
    # # AUC for normalized values
    # print("normalized Auc")
    # auc(x_normalized)
    return x

approaches = [
    # comparison values:
    # "Lime_quickshift_1_7_015",
    # "Lime_slic_80_100_0",
    # "Lime_felzenswalb_71_4e-1_0",
    #
    # "occl_4_0",
    # "occl_4_gray",
    #
    #
    "rise_08_16_3000",
    "rise_08_21_3000",
    "rise_08_22_3000",
    "rise_08_23_3000",
    "rise_08_18_3000",
    #
    # "noise_4_blur",
    # "noise_4_black",
    # "noise_4_blur_rawDiff",



    # "Lime_quickshift_1_7_015",
    # "Lime_slic_80_100_0",
    # "Lime_patches_8_6",
    # "Lime_patches_9_9",
    # "Lime_felzenswalb_71_4e-1_0",

    # "occl_1_0",
    # "occl_2_0",
    # "occl_3_0",
    # "occl_4_0",
    # "occl_5_0",
    # "occl_6_0",
    # "occl_7_0",
    # "occl_8_0",
    # "occl_9_0",
    # "occl_10_0",

    # "rise_09_16_1500",
    # "rise_06_16_3000",
    # "rise_08_16_3000",
    # "rise_08_21_3000",
    #"rise_08_23_3000",


    # "noise_1_blur",
    # "noise_2_blur",
    # "noise_3_blur",
    # "noise_4_blur",
    # "noise_5_blur",
    # "noise_6_blur",
    # "noise_7_blur",
    # "noise_8_blur",
    # "noise_9_blur",
    #
    # "noise_1_black",
    # "noise_2_black",
    # "noise_3_black",
    # "noise_4_black",
    # "noise_5_black",
    # "noise_6_black",
    # "noise_7_black",
    # "noise_8_black",
    # "noise_9_black",

    # "noise_3_blur_rawDiff",
    # "noise_4_blur_rawDiff",
    # "noise_3_black_rawDiff",
    # "noise_4_black_rawDiff",
              ]




# Plot settings
# to generate graph with normalization iterate over x_normalized
data = []
game = "frostbite"
for approach in approaches:
    dir_name_ = os.path.join(game, approach)
    scores = load_scores(dir_name_)
    data.append(scores)

data = np.asarray(data)

sns.set(palette='colorblind', style="whitegrid")
#plt.figure(figsize=(10, 7))
#plt.xlim(-0.1, 1.1)
max = data.max()
min = data.min()
plt.ylim(min - (np.abs(min) * 0.1), max + (np.abs(max) * 0.05))
#plt.xlabel("percentage of deleted pixels")
#plt.ylabel("classification probability")
z = np.linspace(0,1,85)
tick_size = 20
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

i = 0
for appr in data:
    plt.plot(z, appr, label=approaches[i])
    i += 1
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
          ncol=2, mode="expand", borderaxespad=0.)
plt.tight_layout()
#plt.savefig(fname="figures/backup_" + dir_name_ + "/" + dir_name_ + "_insertion.png")
plt.show()









