""" Module for doing a parameter search on different subsets of 10 states of the game Pacman with fast parameter configurations of occlusion.
This can then be used to verify which states are suited to search for good parameters."""

import numpy as np
import os
import re
import keras
import timeit

import applications.atari.rise as rise
from applications.atari.explanation import explainer


def parameter_test(segmentation, ins_color, state_path, model, results_dir):
    """ do a parameter test with only the fast versions of occlusion """
    assert segmentation in ["quickshift", "felzenswalb", "slic", "patches", "rise", "occlusion", "noise"]
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

    if segmentation == "occlusion":
        save_dir = os.path.join(results_dir, "parameter_search")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for i in range(4, 11):
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

    save_list = [best_parameters, times, q_vals]
    np.save(os.path.join(save_dir, "best_parameters_" + ins_color), save_list)


if __name__ == '__main__':

    model_ = keras.models.load_model('../../models/MsPacman_5M_ingame_reward.h5')

    # search for the random states chosen in select_random_states.py
    for i in range(10):
        state_path_ = "random_states_" + str(i) + "/"
        for segmentation_ in ["occlusion"]:
            for ins_color_ in ["black", "random"]:
                parameter_test(segmentation_, ins_color_, state_path=state_path_, model=model_, results_dir=state_path_)

    # search for different HIGHLIGHT-DIV configurations
    for config in ["thres10", "thres20", "thres30", "thres40","pos_neg_thres10", "pos_neg_thres40", "diverse_importance",
                   "thres25", "thres28", "thres32", "thres33", "thres35"]:
        state_path_ = "highlight_states_" + config + "/"
        for segmentation_ in ["occlusion"]:
            for ins_color_ in ["black", "random"]:
                parameter_test(segmentation_, ins_color_, state_path=state_path_, model=model_, results_dir=state_path_)




