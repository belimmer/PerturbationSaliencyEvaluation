""" Module for testing different parameters for the perturbation-based saliency maps."""

import skimage.segmentation as seg
import numpy as np
import os
import re
import keras
import timeit

import applications.atari.rise as rise
from applications.atari.explanation import explainer
from applications.atari.custom_lime import rectangle_segmentation


def test_parameters(_states, _segmentation_fn, parameters, q_vals, _parameters, _times,
                    my_explainer_, insertion_metric, num_samples=1000):
    """
    Helper Function to do the parameter test for LIME segmentation algorithms
    :param _states: the states the should be tested
    :param _segmentation_fn: the segmentation function used by LIME
    :param parameters: the parameters used in the segmenation function
    :param q_vals: the list that stores the q-values during the insertion test, this function adds new values to the list
    :param _parameters: the list that stores the parameters, this function adds new values to the list
    :param _times: the list that stores the time values, this function adds new values to the list
    :param my_explainer_: the explainer used to create the saliency maps
    :param insertion_metric: the insertion metric object which holds for example the information if black or random baseline is used
    :param num_samples: the number of training samples for LIME
    :return: nothing, but q_vals, parameters and times are updated
    """

    time = 0
    for idx in range(len(_states)):
        state = _states[idx]
        input = np.squeeze(state[0])
        start = timeit.default_timer()
        explanation, mask, ranked_mask = my_explainer_.generate_lime_explanation(input=input,
                                                                                hide_img=False,
                                                                                positive_only=True,
                                                                                segmentation_fn= _segmentation_fn,
                                                                                num_samples= num_samples)
        stop = timeit.default_timer()
        time += stop - start
        print("time:" + str(stop - start))
        scores = insertion_metric.single_run(img_tensor=input, explanation=ranked_mask)
        q_vals[idx].append(scores)

    _parameters.append(parameters)
    _times.append(time)


def parameter_test(segmentation, ins_color, state_path, model):
    assert segmentation in ["quickshift", "felzenswalb", "slic", "patches", "rise", "occlusion", "noise", "sarfa"]
    assert ins_color in ['black', 'random']
    if ins_color == "black":
        substrate_function = rise.custom_black
    elif ins_color == "random":
        substrate_function = rise.random_occlusion

    states = []

    for state_name in os.listdir(path=state_path):
        if state_name.endswith(".npy"):
            states.append((np.load(state_path + state_name), re.findall("(\d+)", state_name)[0]))

    my_explainer = explainer(model=model)
    insertion = rise.CausalMetric(model=model, mode='ins', step=np.squeeze(states[0][0]).shape[0],
                                  substrate_fn=substrate_function)

    # reset lists to hold parameter test results
    best_parameters = []
    times = []
    q_vals = []
    for state in states:
        q_vals.append([])

    results_dir = "parameter_results"

    if segmentation == "quickshift":
        # Quckschift
        save_dir = os.path.join(results_dir, "quickshift")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for i in range(0, 6):
            for j in range(1, 5):
                for k in range(0, 4):
                    for r in range(0, 5):
                        kernel_size = 1 + i
                        max_dist = kernel_size * j
                        ratio = k * 0.33
                        num_samples = 1000 + (r * 500)

                        segmentation_fn = (
                            lambda x: seg.quickshift(x, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio,
                                                     convert2lab=False))

                        test_parameters(states, segmentation_fn, (kernel_size, max_dist, ratio, num_samples), q_vals,
                                        best_parameters, times, num_samples=num_samples,
                                        my_explainer_ = my_explainer, insertion_metric = insertion )

        # save the value arrays modified by test parameters
        save_list = [best_parameters, times, q_vals]
        np.save(os.path.join(save_dir, "best_parameters_" + ins_color), save_list)

    if segmentation == "felzenswalb":
        save_dir = os.path.join(results_dir, "felzenswalb")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

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

                        test_parameters(states, segmentation_fn, (scale, sigma, min_size, num_samples), q_vals,
                                        best_parameters, times, num_samples=num_samples,
                                        my_explainer_ = my_explainer, insertion_metric = insertion )

        # save the value arrays modified by test parameters
        save_list = [best_parameters, times, q_vals]
        np.save(os.path.join(save_dir, "best_parameters_" + ins_color), save_list)

    if segmentation == "slic":
        save_dir = os.path.join(results_dir, "slic")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

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

                        test_parameters(states, segmentation_fn, (n_segments, compactness, sigma, num_samples),
                                        q_vals, best_parameters,
                                        times, num_samples=num_samples,
                                        my_explainer_ = my_explainer, insertion_metric = insertion )

        # save the value arrays modified by test parameters
        save_list = [best_parameters, times, q_vals]
        np.save(os.path.join(save_dir, "best_parameters_" + ins_color), save_list)

    if segmentation == "patches":
        save_dir = os.path.join(results_dir, "patches")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for i in range(4, 16):
            for j in range(4, 16):
                segmentation_fn = (
                    lambda x: rectangle_segmentation(x, (i, j)))

                test_parameters(states, segmentation_fn, (i, j), q_vals, best_parameters,
                                        my_explainer_ = my_explainer, insertion_metric=insertion)

            # save the value arrays modified by test parameters
        save_list = [best_parameters, times, q_vals]
        np.save(os.path.join(save_dir, "best_parameters_" + ins_color), save_list)

    if segmentation == "rise":
        save_dir = os.path.join(results_dir, "rise")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for i in range(1, 10):
            for j in range(4, 25):
                for k in range(1, 7):
                    for l in range(0, 2):
                        probability = 0.1 * i
                        mask_size = j
                        number_of_masks = 500 * k
                        softmax = l

                        my_explainer = explainer(model=model)  # need new explainer since the masks are saved

                        time = 0
                        for idx in range(len(states)):
                            state = states[idx]
                            input = np.squeeze(state[0])
                            start = timeit.default_timer()
                            saliency_map = my_explainer.generate_rise_prediction(input,
                                                                                 probability=probability,
                                                                                 use_softmax=softmax,
                                                                                 number_of_mask=number_of_masks,
                                                                                 mask_size=mask_size)
                            stop = timeit.default_timer()
                            time += stop - start
                            print("time:" + str(stop - start))
                            scores = insertion.single_run(img_tensor=input, explanation=saliency_map)
                            q_vals[idx].append(scores)

                        parameters = (probability, mask_size, number_of_masks, softmax)

                        best_parameters.append(parameters)
                        times.append(time)

    if segmentation == "occlusion":
        save_dir = os.path.join(results_dir, "occl")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for i in range(1, 11):
            for j in range(0, 2):
                for k in range(0, 2):
                    patch_size = i
                    color = 0.5 * j
                    softmax = k
                    parameters = (patch_size, color, softmax)

                    time = 0
                    for idx in range(len(states)):
                        state = states[idx]
                        input = np.squeeze(state[0])
                        start = timeit.default_timer()
                        saliency_map = my_explainer.generate_occlusion_explanation(input=input, patch_size=patch_size,
                                                                                   color=color,
                                                                                   use_softmax=softmax)
                        stop = timeit.default_timer()
                        time += stop - start
                        print("time:" + str(stop - start))
                        scores = insertion.single_run(img_tensor=input, explanation=saliency_map)
                        q_vals[idx].append(scores)

                    best_parameters.append(parameters)
                    times.append(time)

    if segmentation == "noise":
        save_dir = os.path.join(results_dir, "noise")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for i in range(1, 11):
            radius = i
            parameters = (radius)

            time = 0
            for idx in range(len(states)):
                state = states[idx]
                input = np.squeeze(state[0])
                start = timeit.default_timer()
                saliency_map = my_explainer.generate_greydanus_explanation(input, r=radius, blur=True)
                stop = timeit.default_timer()
                time += stop - start
                print("time:" + str(stop - start))
                scores = insertion.single_run(img_tensor=input, explanation=saliency_map)
                q_vals[idx].append(scores)

            best_parameters.append(parameters)
            times.append(time)

    if segmentation == "sarfa":
        save_dir = os.path.join(results_dir, "sarfa")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for i in range(1, 11):
            for j in range(0,2):
                radius = i
                blur = j
                parameters = (radius, blur)

                time = 0
                for idx in range(len(states)):
                    state = states[idx]
                    input = np.squeeze(state[0])
                    start = timeit.default_timer()
                    saliency_map = my_explainer.generate_sarfa_explanation(input, r=radius, blur=blur)
                    stop = timeit.default_timer()
                    time += stop - start
                    print("time:" + str(stop - start))
                    scores = insertion.single_run(img_tensor=input, explanation=saliency_map)
                    q_vals[idx].append(scores)

                best_parameters.append(parameters)
                times.append(time)

    save_list = [best_parameters, times, q_vals]
    np.save(os.path.join(save_dir, "best_parameters_" + ins_color), save_list)


if __name__ == '__main__':
    state_path_ = "Verification/highlight_states_thres30/"
    model_ = keras.models.load_model('../models/MsPacman_5M_ingame_reward.h5')

    for segmentation_ in ["occlusion", "noise","rise", "quickshift", "slic", "felzenswalb", "sarfa"]:
        for ins_color_ in ["black", "random"]:
            parameter_test(segmentation_, ins_color_, state_path = state_path_, model = model_)



