""" Module for testing LIME with different segmentation algorithms"""

import skimage.segmentation as seg
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import timeit


def test_quickshift(input, **kwargs):
    test = seg.quickshift(input, convert2lab=False, **kwargs)
    plt.imshow(seg.mark_boundaries(input[:, :, 3], test))
    plt.show()


def test_parameters(_states, _segmentation_fn):
    state = _states[0]
    input = np.squeeze(state[0])
    start = timeit.default_timer()
    test = _segmentation_fn(input)
    stop = timeit.default_timer()
    print("time:" + str(stop - start))
    plt.imshow(seg.mark_boundaries(input[:, :, 3], test))
    plt.show()


if __name__ == '__main__':
    state_path = "HIGHLIGHTS_states/"
    segmentation = "slic"

    states = []

    for state_name in os.listdir(path=state_path):
        if state_name.endswith(".npy"):
            states.append((np.load(state_path + state_name), re.findall("(\d+)", state_name)[0]))

    if segmentation == "quickshift":
        # Quckschift
        save_dir = os.path.join("parameter_results", "quickshift")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        for i in range(0, 6):
            for j in range(1, 5):
                for k in range(1, 4):
                    kernel_size = 1 + i
                    max_dist = j * kernel_size
                    ratio = k * 0.33

                    segmentation_fn = (lambda x : seg.quickshift(x, kernel_size=kernel_size, max_dist= max_dist ,ratio=ratio, convert2lab=False))

                    test_parameters(states, segmentation_fn)

    if segmentation == "felzenswalb":
        save_dir = os.path.join("parameter_results", "felzenswalb")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        for i in range(0, 6):
            for j in range(0, 5):
                for k in range(0, 6):
                    scale = 1 + 20 * i
                    sigma = j * 0.25
                    min_size = 4 * k

                    segmentation_fn = (
                        lambda x: seg.felzenszwalb(x, scale=scale, sigma=sigma, min_size=min_size))

                    test_parameters(states, segmentation_fn)

    if segmentation == "slic":
        save_dir = os.path.join("parameter_results", "slic")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        for i in range(0, 6):
            for j in range(0, 5):
                for k in range(0, 5):
                    n_segments = 40 + (i * 40)
                    sigma = j * 0.25
                    compactness = 0.001 * (10**k)

                    segmentation_fn = (
                        lambda x: seg.slic(x, n_segments=n_segments, compactness=compactness, sigma=sigma))

                    test_parameters(states, segmentation_fn)
