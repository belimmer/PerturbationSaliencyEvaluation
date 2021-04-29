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


def test_quickshift(input, **kwargs):
    test = seg.quickshift(input, convert2lab=False, **kwargs)
    plt.imshow(seg.mark_boundaries(input[:, :, 3], test))
    plt.show()


def test_parameters(_states, _segmentation_fn, parameters, _best_aucs, _best_parameters, _save_dir):
    sum = 0
    for state in _states:
        input = np.squeeze(state[0])
        start = timeit.default_timer()
        explanation, mask, ranked_mask = my_explainer.generate_lime_explanation(input=input,
                                                                                hide_img=False,
                                                                                positive_only=True,
                                                                                segmentation_fn= _segmentation_fn)
        score = insertion.single_run(img_tensor=input, explanation=ranked_mask, name=state[1],
                                     approach="lime", use_softmax=True, plot=False)
        stop = timeit.default_timer()
        print("time:" + str(stop - start))
        auc = score.sum() / (score.shape[0])
        sum += auc

    if len(_best_aucs) < 5:
        _best_aucs.append(sum)
        _best_parameters.append(parameters)
        # sort the lists
        best_parameters = [x for (_, x) in
                           sorted(zip(_best_aucs, _best_parameters), reverse=True, key=lambda pair: pair[0])]
        best_aucs = sorted(_best_aucs, reverse=True)
        # update files
        np.savetxt(os.path.join(_save_dir, "best_parameters.txt"), best_parameters)
        np.savetxt(os.path.join(_save_dir, "best_aucs.txt"), best_aucs)
    else:
        for idx in range(5):
            if _best_aucs[idx] < sum:
                # add new values
                _best_aucs.insert(idx, sum)
                _best_parameters.insert(idx, parameters)
                # remove old values
                _best_aucs.pop(-1)
                _best_parameters.pop(-1)
                # update files
                np.savetxt(os.path.join(_save_dir, "best_parameters.txt"), _best_parameters)
                np.savetxt(os.path.join(_save_dir, "best_aucs.txt"), _best_aucs)
                break


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
        for i in range(0, 10):
            for j in range(0, 10):
                for k in range(1, 7):
                    kernel_size = 1 + (i * 0.2)
                    max_dist = 1 + j
                    ratio = k * 0.15

                    segmentation_fn = (lambda x : seg.quickshift(x, kernel_size=kernel_size, max_dist= max_dist ,ratio=ratio, convert2lab=False))

                    test_parameters(states, segmentation_fn, (kernel_size,max_dist, ratio), best_aucs, best_parameters, save_dir)

    if segmentation == "felzenswalb":
        save_dir = os.path.join("parameter_results", "felzenswalb")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        for i in range(0, 10):
            for j in range(0, 10):
                for k in range(0, 7):
                    scale = 1 + (i * 10)
                    sigma = j * 0.1
                    min_size = k * 5

                    segmentation_fn = (
                        lambda x: seg.felzenszwalb(x, scale=scale, sigma=sigma, min_size=min_size))

                    test_parameters(states, segmentation_fn, (scale, sigma, min_size), best_aucs, best_parameters,
                                    save_dir)

    if segmentation == "slic":
        save_dir = os.path.join("parameter_results", "slic")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        best_aucs = []
        best_parameters = []
        for i in range(0, 21):
            for j in range(0, 11):
                for k in range(0, 5):
                    n_segments = 40 + (i * 20)
                    sigma = j * 0.1
                    compactness = 0.01 * (10**k)

                    segmentation_fn = (
                        lambda x: seg.slic(x, n_segments=n_segments, compactness=compactness, sigma=sigma))

                    test_parameters(states, segmentation_fn, (n_segments, compactness, sigma), best_aucs, best_parameters,
                                   save_dir)

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
        for i in range(1, 10):
            for j in range(4, 25):
               for k in range(1,7):
                   probability = 0.1 * i
                   mask_size = j
                   number_of_masks = 500 * k

                   my_explainer = explainer(model=model) # need new explainer since the masks are saved

                   sum = 0
                   for state in states:
                       input = np.squeeze(state[0])
                       start = timeit.default_timer()
                       saliency_map = my_explainer.generate_rise_prediction(input,
                                                                            probability=probability,
                                                                            use_softmax = True,
                                                                            number_of_mask = number_of_masks,
                                                                            mask_size=mask_size)
                       score = insertion.single_run(img_tensor=input, explanation=saliency_map, name=state[1],
                                                    approach="lime", use_softmax=True, plot=False)
                       stop = timeit.default_timer()
                       print("time:" + str(stop - start))
                       auc = score.sum() / (score.shape[0])
                       sum += auc

                   parameters=(probability,mask_size,number_of_masks)
                   if len(best_aucs) < 5:
                       best_aucs.append(sum)
                       best_parameters.append(parameters)
                       # sort the lists
                       best_parameters = [x for (_, x) in
                                          sorted(zip(best_aucs, best_parameters), reverse=True,
                                                 key=lambda pair: pair[0])]
                       best_aucs = sorted(best_aucs, reverse=True)
                       # update files
                       np.savetxt(os.path.join(save_dir, "best_parameters.txt"), best_parameters)
                       np.savetxt(os.path.join(save_dir, "best_aucs.txt"), best_aucs)
                   else:
                       for idx in range(5):
                           if best_aucs[idx] < sum:
                               # add new values
                               best_aucs.insert(idx, sum)
                               best_parameters.insert(idx, parameters)
                               # remove old values
                               best_aucs.pop(-1)
                               best_parameters.pop(-1)
                               # update files
                               np.savetxt(os.path.join(save_dir, "best_parameters.txt"), best_parameters)
                               np.savetxt(os.path.join(save_dir, "best_aucs.txt"), best_aucs)
                               break
