"""
This module was adapted from a module in
https://github.com/greydanus/visualize_atari
Date: 2020
commit: 81c9248543431ec3925b982a9f464128e7135677
License: MIT
"""

import numpy as np
from PIL import Image
from sklearn.utils.extmath import softmax


class custom_greydanus_explainer():
    """
    This is the modified version of the greydanus explainer. Instead of occluding images using a gaussian blur,
    a black shape is used.
    """
    occlude = lambda I, mask: I * (1 - mask)

    def get_mask(self, center, size, radius):
        """
        Creates a mask to occlude the image

        Args:
            center: center position of the mask
            size: size of the mask

        Returns:
            mask: The newly created mask

        """
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        # distance to center(calculated with pythagoras) has to be lower then or equal to radius
        keep = x * x + y * y <= radius * radius
        mask = np.zeros(size)
        mask[keep] = 1  # select a circle of pixels
        return mask

    def generate_explanation(self, stacked_frames, model, radius, raw_diff = False, neuron_selection = False):
        """
        Generates an explanation for the prediction of a CNN

        Args:
            stacked_frames: input of which the prediction will be explained
            model: the model which should be explained
            radius: the radius of the black circle

        Returns:
            scores: The saliency map which functions as explanantion
        """
        # d: density of scores (if d==1, then get a score for every pixel...
        #    if d==2 then every other, which is 25% of total pixels for a 2D image)
        d = radius

        my_input = np.expand_dims(stacked_frames, axis=0)
        original_output = model.predict(my_input)

        x = stacked_frames.shape[0]
        y = stacked_frames.shape[1]

        scores = np.zeros((int((x-1) / d) + 1, int((y-1) / d) + 1))  # saliency scores S(t,i,j)

        for i in range(0, x, d):
            for j in range(0, y, d):
                mask = self.get_mask(center=[i, j], size=[x, y], radius=radius)
                stacked_mask = np.zeros(shape=stacked_frames.shape)
                for idx in range(stacked_frames.shape[2]):
                    stacked_mask[:, :, idx] = mask

                masked_input = np.expand_dims(custom_greydanus_explainer.occlude(stacked_frames, stacked_mask), axis=0)
                masked_output = model.predict(masked_input)
                if raw_diff:
                    if neuron_selection is not False:
                        action_index = neuron_selection
                    else:
                        action_index = np.argmax(original_output)
                    scores[int(i / d), int(j / d)] = 1 - np.squeeze(softmax(masked_output))[action_index]
                else:
                    scores[int(i / d), int(j / d)] = (pow(original_output - masked_output, 2).sum() * 0.5)

        pmax = scores.max()
        scores = Image.fromarray(scores).resize(size=[x, y], resample=Image.BILINEAR)
        scores = pmax * scores / np.array(scores).max()
        return scores
