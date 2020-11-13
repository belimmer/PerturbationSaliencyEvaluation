import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Normalizing formula from https://arxihttps://arxiv.org/pdf/2001.00396.pdfv.org/pdf/2001.00396.pdf
# did not use this sicne we wanted to show the confidence range
def normalize(arr):
    out = []
    b = arr[0]
    t_1 = arr[-1]
    for val in arr:
        out.append((val - b)/(t_1 - b))
    out = np.asarray(out)
    return out


approaches = ["Occlusion Sensitivity Original", "Noise Sensitiviy Black", "Noise Sensitiviy Blur", "LIME", "RISE", "Random"]
# color = ["y","c","r","k","g", "b"] better to use sns colorblind
# In case you one only wants to look at certain approaches
ignored_index = [5]

env = "perdictions"


# needs to be defined here since we need approaches
def auc(data):
    """
    simple Area Under the curve calculation using sum_of_values/number_of_values
    :param data:
    :return:
    """
    i = 0
    for arr in data:
        if i in ignored_index:
            i += 1
            continue
        print(approaches[i])
        auc = arr.sum() / (arr.shape[0])
        print(auc)
        i += 1


# load the different data checkpoints
x_0 = np.load(file="figures/backup_" + env + "/pred_100.npy")
x_1 = np.load(file="figures/backup_" + env + "/pred_200.npy")
x_2 = np.load(file="figures/backup_" + env + "/pred_300.npy")
x_3 = np.load(file="figures/backup_" + env + "/pred_400.npy")
x_4 = np.load(file="figures/backup_" + env + "/pred_500.npy")
x_5 = np.load(file="figures/backup_" + env + "/pred_600.npy")
x_6 = np.load(file="figures/backup_" + env + "/pred_700.npy")
x_7 = np.load(file="figures/backup_" + env + "/pred_800.npy")
x_8 = np.load(file="figures/backup_" + env + "/pred_900.npy")
x_9 = np.load(file="figures/backup_" + env + "/pred_1000.npy")

# transpose the data arrays such that the first dimension corrsponds to the approaches (e.g. x_0[0] is the data for
# Occlusion Sensitivity
x_0 = np.transpose(x_0, (1,2,0))
x_1 = np.transpose(x_1, (1,2,0))
x_2 = np.transpose(x_2, (1,2,0))
x_3 = np.transpose(x_3, (1,2,0))
x_4 = np.transpose(x_4, (1,2,0))
x_5 = np.transpose(x_5, (1,2,0))
x_6 = np.transpose(x_6, (1,2,0))
x_7 = np.transpose(x_7, (1,2,0))
x_8 = np.transpose(x_8, (1,2,0))
x_9 = np.transpose(x_9, (1,2,0))

# combine the data checkpoints to one array
x = np.concatenate((x_0, x_1), axis=2)
x = np.concatenate((x, x_2), axis=2)
x = np.concatenate((x, x_3), axis=2)
x = np.concatenate((x, x_4), axis=2)
x = np.concatenate((x, x_5), axis=2)
x = np.concatenate((x, x_6), axis=2)
x = np.concatenate((x, x_7), axis=2)
x = np.concatenate((x, x_8), axis=2)
x = np.concatenate((x, x_9), axis=2)
# average the values for each step of the insertion game over all tested inputs
number = x.shape[2]
x = np.sum(x, axis=2)
x = x / number

# AUC for raw softmax values
auc(x)

# normalize prediction values to have minimum 0 and maximum 1
x_normalized = []
for appr in x:
    x_normalized.append(normalize(appr))

# AUC for normalized values
print("normalized Auc")
auc(x_normalized)


# Plot settings
# to generate graph with normalization iterate over x_normalized
data = x
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
    if i in ignored_index:
        i += 1
        continue
    plt.plot(z, appr, label=approaches[i])
    i += 1
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#           ncol=2, mode="expand", borderaxespad=0.)
plt.tight_layout()
plt.savefig(fname="figures/backup_" + env + "/" + env + "_insertion.png")
plt.show()









