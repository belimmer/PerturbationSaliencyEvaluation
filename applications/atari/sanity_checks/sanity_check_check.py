"""
module for testing the similarity metrics on randomly generated saliency maps
"""

from applications.atari.sanity_checks.sanity_checks_main import calc_sim
import numpy as np
import matplotlib.pyplot as plt


####### Calculate mean similartiy for UNIFORMLY sampled saliency maps
pearson_list = []
ssim_list = []
spearman_list = []

for i in range(100):
    test = np.random.rand(84,84)
    # plt.imshow(test)
    # plt.show()

    test2 = np.random.rand(84,84)
    # plt.imshow(test2)
    # plt.show()

    calc_sim(test,test2, pearson_list, ssim_list, spearman_list)

print("UNIFORM:")
print("pearson")
print(np.mean(pearson_list))
print(np.std(pearson_list))
print("ssim")
print(np.mean(ssim_list))
print(np.std(ssim_list))
print("spearman")
print(np.mean(spearman_list))

########## Calculate mean similarity values for saliency maps with gaussian distribution
pearson_list = []
ssim_list = []
spearman_list = []

for i in range(100):
    test = np.random.normal(size=(84, 84))
    #test = np.random.normal(loc=0.5, scale=0.5,size=(84,84))
    # plt.imshow(test)
    # plt.show()

    test2 = np.random.normal(size=(84, 84))
    #test2 = np.random.normal(loc=0.5, scale=0.5,size=(84,84))
    # plt.imshow(test2)
    # plt.show()

    calc_sim(test,test2, pearson_list, ssim_list, spearman_list)

print("Gaussian:")
print("pearson")
print(np.mean(pearson_list))
print(np.std(pearson_list))
print("ssim")
print(np.mean(ssim_list))
print(np.std(ssim_list))
print("spearman")
print(np.mean(spearman_list))



