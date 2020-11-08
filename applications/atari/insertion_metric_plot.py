import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def normalize(arr):
    out = []
    b = arr[0]
    t_1 = arr[-1]
    for val in arr:
        out.append((val - b)/(t_1 - b))
    out = np.asarray(out)
    return out


#approaches = ["Occlusion Sensitivity Original", "Occlusion Sensitivity Modified", "Noise Sensitiviy Black", "Noise Sensitiviy Blur", "LIME", "RISE", "Random"]
approaches = ["Occlusion Sensitivity Original", "Noise Sensitiviy Black", "Noise Sensitiviy Blur", "LIME", "RISE", "Random"]
# color = ["y","c","r","k","g", "b"] better to use sns colorblind palette
ignored_index = [1]

env = "space_invaders"

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

x = np.concatenate((x_0, x_1), axis=2)
x = np.concatenate((x, x_2), axis=2)
x = np.concatenate((x, x_3), axis=2)
x = np.concatenate((x, x_4), axis=2)
x = np.concatenate((x, x_5), axis=2)
x = np.concatenate((x, x_6), axis=2)
x = np.concatenate((x, x_7), axis=2)
x = np.concatenate((x, x_8), axis=2)
x = np.concatenate((x, x_9), axis=2)
number = x.shape[2]
x = np.sum(x, axis=2)
x = x / number

# needs to be defined here since we need approaches
def auc(data):
    i = 0
    for arr in data:
        if i in ignored_index:
            i += 1
            continue
        print(approaches[i])
        auc = arr.sum() / (arr.shape[0])
        print(auc)
        i += 1

auc(x)

x_normalized = []
for appr in x:
    x_normalized.append(normalize(appr))

print("normalized Auc")
auc(x_normalized)

# to generate graph without normalization iterate over x
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









