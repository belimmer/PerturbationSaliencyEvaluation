"""
This module contains an inplementation of SARFA as described in https://arxiv.org/pdf/1912.12191.pdf

The code is derived from
https://github.com/nikaashpuri/sarfa-saliency/blob/master/visualize_atari/saliency.py
Date: 2021
commit: 5797e32a5a7f7a997361de202551687c667d06aa
"""

import numpy as np
from PIL import Image
from scipy.special import softmax
from scipy.stats import entropy

from scipy.ndimage.filters import gaussian_filter


def cross_entropy(original_output, perturbed_output, action_index):
    # remove the chosen action in original output
    p = original_output[:action_index]
    p = np.append(p, original_output[action_index + 1:])
    # According to equation (2) in the paper(https://arxiv.org/abs/1912.12191v4)
    # the softmax should happen over the out put with the chosen action removed.
    # We do it like this here but we want to mention that this differs from the
    # implementation in https://github.com/nikaashpuri/sarfa-saliency/blob/master/visualize_atari/saliency.py
    p = softmax(p)

    # Do the same for the perturbed output
    new_p = perturbed_output[:action_index]
    new_p = np.append(new_p, perturbed_output[action_index + 1:])
    new_p = softmax(new_p)

    # According to the paper this should be the other way around: entropy(new_p,p)
    # (directly und er equation (2) in https://arxiv.org/pdf/1912.12191.pdf)
    # While this would make a difference, it is like this in the official implementation in
    # github.com/nikaashpuri/sarfa-saliency/blob/master/visualize_atari/saliency.py:
    KL = entropy(p, new_p)

    K = 1. / (1. + KL)

    return K


def sarfa_saliency(original_output, perturbed_output, action_index):
    """
    Calculate the impact of the perturbed area in *perturbed_output* for the action *action_index*
    according to the SARFA formula.
    """
    original_output = np.squeeze(original_output)
    perturbed_output = np.squeeze(perturbed_output)
    dP = softmax(original_output)[action_index] - softmax(perturbed_output)[action_index]
    if dP > 0:
        K = cross_entropy(original_output, perturbed_output, action_index)
        return (2 * K * dP) / (K + dP)
    else:
        return 0


class SarfaExplainer():
    """
    For cerating saliency maps using SARFA (https://arxiv.org/pdf/1912.12191.pdf)
    """

    occlude = lambda I, mask: I * (1 - mask)
    occlude_blur = lambda I, mask: I * (1 - mask) + gaussian_filter(I, sigma=3) * mask # choose an area to blur

    def get_occlusion_mask(self, center, size, radius):
        """
        Creates a mask to occlude the image with black color

        Args:
            center: center position of the mask
            size: size of the mask
            radius: the radius of the mask

        Returns:
            mask: The newly created mask

        """
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        # distance to center(calculated with pythagoras) has to be lower then or equal to radius
        keep = x * x + y * y <= radius * radius
        mask = np.zeros(size)
        mask[keep] = 1  # select a circle of pixels
        return mask

    def get_blur_mask(self, center, size, radius):
        """
        Creates the blurred mask, which will be added to the  image.

        Args:
            center: center position of mask
            size: size of the mask
            radius: radius of the blurring

        Returns:
            mask: mask which is used to perturb images
        """
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        keep = x * x + y * y <= 1
        mask = np.zeros(size)
        mask[keep] = 1  # select a circle of pixels
        mask = gaussian_filter(mask, sigma=radius)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
        return mask / mask.max()

    def generate_explanation(self, stacked_frames, model, radius, blur, neuron_selection=False):
        """
        Generates a SARFA explanation for the prediction of a CNN

        Args:
            stacked_frames: input of which the prediction will be explained
            model: the model which should be explained
            radius: the radius of the black circle
            blur: use blur or occlusion for perturbation
            neuron_selection: if False, the best action is explained
                            otherwise this should be the index of the action to be explained

        Returns:
            scores: The saliency map which functions as explanantion
        """
        # d: density of scores (if d==1, then get a score for every pixel...
        #    if d==2 then every other, which is 25% of total pixels for a 2D image)
        d = radius

        my_input = np.expand_dims(stacked_frames, axis=0)
        original_output = model.predict(my_input)

        # get the action to be explained
        if neuron_selection is not False:
            action_index = neuron_selection
        else:
            action_index = np.argmax(original_output)

        x = stacked_frames.shape[0]
        y = stacked_frames.shape[1]

        scores = np.zeros((int((x-1) / d) + 1, int((y-1) / d) + 1))  # saliency scores S(t,i,j)

        for i in range(0, x, d):
            for j in range(0, y, d):
                if blur:
                    mask = self.get_blur_mask(center=[i, j], size=[x, y], radius=radius)
                else:
                    mask = self.get_occlusion_mask(center=[i, j], size=[x, y], radius=radius)
                stacked_mask = np.zeros(shape=stacked_frames.shape)
                for idx in range(stacked_frames.shape[2]):
                    stacked_mask[:, :, idx] = mask

                if blur:
                    masked_input = np.expand_dims(SarfaExplainer.occlude_blur(stacked_frames, stacked_mask), axis=0)
                else:
                    masked_input = np.expand_dims(SarfaExplainer.occlude(stacked_frames, stacked_mask), axis=0)
                masked_output = model.predict(masked_input)

                scores[int(i / d), int(j / d)] = sarfa_saliency(original_output, masked_output, action_index)

        pmax = scores.max()
        scores = Image.fromarray(scores).resize(size=[x, y], resample=Image.BILINEAR)
        scores = pmax * scores / np.array(scores).max()
        return scores
