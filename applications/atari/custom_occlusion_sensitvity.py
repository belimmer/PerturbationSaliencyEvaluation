"""
This module was adapted from a module in
https://github.com/sicara/tf-explain
Date: 2020
commit: 8dff129ff7b1012dba2761a61e3c3e68e9ecbec2
License: MIT
"""


import math

import cv2
import numpy as np

from tf_explain.utils.display import grid_display, heatmap_display
from tf_explain.utils.saver import save_rgb
from sklearn.utils.extmath import softmax


class CustomOcclusionSensitivity:

    """
    Perform Occlusion Sensitivity for a given input.
    """

    def __init__(self, batch_size=None):
        self.batch_size = batch_size

    def explain(
        self,
        validation_data,
        model,
        class_index,
        patch_size,
        colormap=cv2.COLORMAP_VIRIDIS,
    ):
        """
        Compute Occlusion Sensitivity maps for a specific class index.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            patch_size (int): Size of patch to apply on the image
            colormap (int): OpenCV Colormap to use for heatmap visualization

        Returns:
            np.ndarray: Grid of all the sensitivity maps with shape (batch_size, H, W, 3)
        """
        images, _ = validation_data
        sensitivity_maps = np.array(
            [
                self.get_sensitivity_map(model, image, class_index, patch_size, color=0.0)
                for image in images
            ]
        )

        heatmaps = np.array(
            [
                heatmap_display(heatmap, image, colormap)
                for heatmap, image in zip(sensitivity_maps, images)
            ]
        )

        grid = grid_display(heatmaps)

        return grid

    def get_sensitivity_map(self, model, image, class_index, patch_size, color, use_softmax=False, use_old_confidence=False):
        """
        Compute sensitivity map on a given image for a specific class index.

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            image:
            class_index (int): Index of targeted class
            patch_size (int): Size of patch to apply on the image
            use_softmax (bool): should a softmax be used on the predictions of the model
            use_old_confidence (bool): compute the saliency map with original_confidence - prediction, or 1 - prediction

        Returns:
            np.ndarray: Sensitivity map with shape (H, W, 3)
        """
        sensitivity_map = np.zeros(
            (
                math.ceil(image.shape[0] / patch_size),
                math.ceil(image.shape[1] / patch_size),
            )
        )

        patches = [
            custom_apply_grey_patch(image, top_left_x, top_left_y, patch_size, color=color)
            for index_x, top_left_x in enumerate(range(0, image.shape[0], patch_size))
            for index_y, top_left_y in enumerate(range(0, image.shape[1], patch_size))
        ]

        coordinates = [
            (index_y, index_x)
            for index_x in range(
                sensitivity_map.shape[1]
            )
            for index_y in range(
                sensitivity_map.shape[0]
            )
        ]

        pred = model.predict(np.expand_dims(image, axis=0))
        if use_softmax:
            pred = softmax(pred)
        original_pred = np.squeeze(pred)[class_index]
        for (index_y, index_x), patch in zip(
            coordinates, patches
        ):
            prediction = model.predict(np.expand_dims(patch, axis=0))
            if use_softmax:
                confidence = np.squeeze(softmax(prediction))[class_index]
            else:
                confidence = np.squeeze(prediction)[class_index]
            if use_old_confidence:
                sensitivity_map[index_y, index_x] = original_pred - confidence
            else:
                sensitivity_map[index_y, index_x] = 1 - confidence

        return cv2.resize(sensitivity_map, image.shape[0:2])

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Grid of all heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_rgb(grid, output_dir, output_name)


def custom_apply_grey_patch(image, top_left_x, top_left_y, patch_size, color):
    """
    Adds a grey patch to the image.

    Args:
        image: image to which the grey patch is added
        top_left_x (int): top left x coordinate of the grey patch
        top_left_y (int): top left y coordinate of the grey patch
        patch_size (int): width and height of the grey patch

    Returns:
        patched_image: image with the added patch
    """
    patched_image = np.array(image, copy=True)
    # 0.0 is the color the occlusion is done with. In the original paper grey was chosen as default value, but
    # black seems to work better in some scenarios.
    patched_image[
    top_left_y: top_left_y + patch_size, top_left_x: top_left_x + patch_size, :
    ] = color

    return patched_image
