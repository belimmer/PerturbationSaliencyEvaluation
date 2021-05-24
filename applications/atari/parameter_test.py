""" Module for testing LIME with different segmentation algorithms"""

import skimage.segmentation as seg
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import keras
import timeit

import applications.atari.rise as rise
from applications.atari.explanation import explainer
from applications.atari.custom_lime import rectangle_segmentation

import pandas as pd


def test_quickshift(input, **kwargs):
    test = seg.quickshift(input, convert2lab=False, **kwargs)
    plt.imshow(seg.mark_boundaries(input[:, :, 3], test))
    plt.show()


def test_parameters(_states, _segmentation_fn, parameters, _best_aucs, _best_parameters, _times, _save_dir, num_samples=1000):
    sum = 0
    time = 0
    for state in _states:
        input = np.squeeze(state[0])
        start = timeit.default_timer()
        explanation, mask, ranked_mask = my_explainer.generate_lime_explanation(input=input,
                                                                                hide_img=False,
                                                                                positive_only=True,
                                                                                segmentation_fn= _segmentation_fn,
                                                                                num_samples= num_samples)
        stop = timeit.default_timer()
        time += stop - start
        print("time:" + str(stop - start))
        score = insertion.single_run(img_tensor=input, explanation=ranked_mask, name=state[1],
                                     approach="lime", use_softmax=True, plot=False)
        auc = score.sum() / (score.shape[0])
        sum += auc

    _best_aucs.append(sum)
    _best_parameters.append(parameters)
    _times.append(time)

    data_frame = pd.DataFrame()
    data_frame["aucs"] = _best_aucs
    data_frame["params"] = _best_parameters
    data_frame["time"] = _times

    data_frame.to_csv(os.path.join(_save_dir, "best_parameters.csv"))

if __name__ == '__main__':
    state_path = "HIGHLIGHTS_states/"
    state_output_path = "output_highlight_states/"
    model = keras.models.load_model('models/MsPacman_5M_ingame_reward.h5')
    segmentation = "rise"

    states = []

    for state_name in os.listdir(path=state_path):
        if state_name.endswith(".npy"):
            states.append((np.load(state_path + state_name), re.findall("(\d+)", state_name)[0]))

    my_explainer = explainer(model=model)
    insertion = rise.CausalMetric(model=model, mode='ins', step=np.squeeze(states[0][0]).shape[0],
                                  substrate_fn=rise.custom_black)

    if segmentation == "quickshift":
        # Quckschift
        save_dir = os.path.join("parameter_results", "quickshift")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        times = []
        for i in range(0, 6):
            for j in range(1, 5):
                for k in range(0, 4):
                    for r in range(0,5):
                        kernel_size = 1 + i
                        max_dist = kernel_size * j
                        ratio = k * 0.33
                        num_samples = 1000 + (r*500)

                        segmentation_fn = (lambda x : seg.quickshift(x, kernel_size=kernel_size, max_dist= max_dist ,ratio=ratio, convert2lab=False))

                        test_parameters(states, segmentation_fn, (kernel_size,max_dist, ratio, num_samples), best_aucs,
                                        best_parameters, times, save_dir, num_samples=num_samples)

    if segmentation == "felzenswalb":
        save_dir = os.path.join("parameter_results", "felzenswalb")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        times = []
        for i in range(0, 6):
            for j in range(0, 5):
                for k in range(0, 9):
                    for r in range(0, 5):
                        scale = 1 + (20 * i)
                        sigma = j * 0.25
                        min_size = k
                        num_samples = 1000 + (r * 500)

                        segmentation_fn = (
                            lambda x: seg.felzenszwalb(x, scale=scale, sigma=sigma, min_size=min_size))

                        test_parameters(states, segmentation_fn, (scale, sigma, min_size, num_samples), best_aucs, best_parameters, times,
                                        save_dir, num_samples= num_samples)

    if segmentation == "slic":
        save_dir = os.path.join("parameter_results", "slic")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        times = []
        for i in range(0, 6):
            for j in range(0, 5):
                for k in range(0, 5):
                    for r in range(0, 5):
                        n_segments = 40 + (i * 40)
                        sigma = j * 0.25
                        compactness = 0.001 * (10 ** k)
                        num_samples = 1000 + (r * 500)

                        segmentation_fn = (
                            lambda x: seg.slic(x, n_segments=n_segments, compactness=compactness, sigma=sigma))

                        test_parameters(states, segmentation_fn, (n_segments, compactness, sigma, num_samples), best_aucs, best_parameters,
                                       times, save_dir, num_samples=num_samples)

    if segmentation == "patches":
        save_dir = os.path.join("parameter_results", "patches")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        for i in range(4, 16):
            for j in range(4, 16):

                    segmentation_fn = (
                        lambda x: rectangle_segmentation(x, (i,j)))

                    test_parameters(states, segmentation_fn, (i,j), best_aucs, best_parameters, save_dir)

    if segmentation == "rise":
        save_dir = os.path.join("parameter_results", "rise")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        times = []
        for i in range(1, 10):
            for j in range(4, 25):
                for k in range(1,7):
                    probability = 0.1 * i
                    mask_size = j
                    number_of_masks = 500 * k

                    my_explainer = explainer(model=model) # need new explainer since the masks are saved

                    sum = 0
                    time = 0
                    for state in states:
                        input = np.squeeze(state[0])
                        start = timeit.default_timer()
                        saliency_map = my_explainer.generate_rise_prediction(input,
                                                                            probability=probability,
                                                                            use_softmax = True,
                                                                            number_of_mask = number_of_masks,
                                                                            mask_size=mask_size)
                        stop = timeit.default_timer()
                        time += stop - start
                        print("time:" + str(stop - start))
                        score = insertion.single_run(img_tensor=input, explanation=saliency_map, name=state[1],
                                                    approach="lime", use_softmax=True, plot=False)
                        auc = score.sum() / (score.shape[0])
                        sum += auc

                    parameters=(probability,mask_size,number_of_masks)
                    best_aucs.append(sum)
                    best_parameters.append(parameters)
                    times.append(time)

                    data_frame = pd.DataFrame()
                    data_frame["aucs"] = best_aucs
                    data_frame["params"] = best_parameters
                    data_frame["time"] = times

                    data_frame.to_csv(os.path.join(save_dir, "best_parameters.csv"))

    if segmentation == "occlusion":
        save_dir = os.path.join("parameter_results", "occl")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        times = []
        for i in range(1, 11):
            for j in range(0,2):
                patch_size = i
                color = 0.5 * j
                parameters = (patch_size, color)

                sum = 0
                time = 0
                for state in states:
                    input = np.squeeze(state[0])
                    start = timeit.default_timer()
                    saliency_map = my_explainer.generate_occlusion_explanation(input=input, patch_size=patch_size, color=color,
                                                                           use_softmax=True)
                    stop = timeit.default_timer()
                    time += stop - start
                    print("time:" + str(stop - start))
                    score = insertion.single_run(img_tensor=input, explanation=saliency_map, name=state[1],
                                                 approach="not_used", use_softmax=True, plot=False)
                    auc = score.sum() / (score.shape[0])
                    sum += auc

                best_aucs.append(sum)
                best_parameters.append(parameters)
                times.append(time)

                data_frame = pd.DataFrame()
                data_frame["aucs"] = best_aucs
                data_frame["params"] = best_parameters
                data_frame["time"] = times

                data_frame.to_csv(os.path.join(save_dir, "best_parameters.csv"))

    if segmentation == "noise":
        save_dir = os.path.join("parameter_results", "noise")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        times = []
        for i in range(1, 11):
            radius = i
            parameters = (radius)

            sum = 0
            time = 0
            for state in states:
                input = np.squeeze(state[0])
                start = timeit.default_timer()
                saliency_map = my_explainer.generate_greydanus_explanation(input, r=radius, blur=True)
                stop = timeit.default_timer()
                time += stop - start
                print("time:" + str(stop - start))
                score = insertion.single_run(img_tensor=input, explanation=saliency_map, name=state[1],
                                             approach="not_used", use_softmax=True, plot=False)
                auc = score.sum() / (score.shape[0])
                sum += auc

            best_aucs.append(sum)
            best_parameters.append(parameters)
            times.append(time)

            data_frame = pd.DataFrame()
            data_frame["aucs"] = best_aucs
            data_frame["params"] = best_parameters
            data_frame["time"] = times

            data_frame.to_csv(os.path.join(save_dir, "best_parameters.csv"))


