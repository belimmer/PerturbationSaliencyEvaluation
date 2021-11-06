"""
This module was adapted from a module in
https://github.com/eclique/RISE
Date: 2020
commit: d91ea006d4bb9b7990347fe97086bdc0f5c1fe10
License: MIT
"""

from tqdm import tqdm
from skimage.transform import resize
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


class rise_explainer():
    """
    Class used to produce RISE explanations.
    """

    def __init__(self, N, s, p1, batch_size):
        self.N = N
        self.s = s
        self.p1 = p1
        self.batch_size = batch_size

    def generate_masks(self, input_size):
        """
        Generates masks which can be used to occlude images.

        Args:
             input_size: Size of the original image
        """
        cell_size = np.ceil(np.array(input_size) / self.s)
        up_size = (self.s + 1) * cell_size

        grid = np.random.rand(self.N, self.s, self.s) < self.p1
        grid = grid.astype('float32')

        masks = np.empty((self.N, *input_size))

        for i in tqdm(range(self.N), desc='Generating masks'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
        masks = masks.reshape(-1, *input_size, 1)
        return masks

    def explain(self, model, inp, masks, input_size, use_softmax):
        """
        Generates an explanation the prediction of a CNN

        Args:
            model: the model which will be explained
            inp: the input image for which the explanation is created
            masks: masks to occlude the image
            input_size: size of the image

        Returns:
            sal: saliency map which functions as a explanation
        """
        predictions = []
        masked = inp * masks
        for i in tqdm(range(0, self.N, self.batch_size), desc='Explaining'):
            preds = model.predict(masked[i:min(i + self.batch_size, self.N)])
            if use_softmax:
                predictions.append(softmax(preds))
            else:
                predictions.append(preds)
        predictions = np.concatenate(predictions)
        sal = predictions.T.dot(masks.reshape(self.N, -1)).reshape(-1, *input_size)
        sal = sal / self.N / self.p1
        return sal


def custom_black(img):
    return np.zeros(shape=[img.shape[0], img.shape[1], img.shape[2]])


def custom_blur(img):
    """Returns blurred version of the image."""
    return gaussian_filter(img, sigma=8)


def random_occlusion(img):
    """ Returns a version of the image that is filled with random floats between 0 and 1"""
    return np.random.uniform(low=0.0, high=1.0, size=[img.shape[0], img.shape[1], img.shape[2]])


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, explanation):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
        Return:
            scores (nd.array): Array containing q-vals at every step.
        """
        HW = img_tensor.shape[0] * img_tensor.shape[1]
        z = img_tensor.shape[2]
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            # 'Pixels deleted'
            start = img_tensor
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            # 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor

        scores = []
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)
        for i in range(n_steps+1):
            pred = self.model.predict(np.expand_dims(start, axis=0))
            scores.append(np.squeeze(pred))

            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start_tmp = start.reshape(HW, z)
                finish_tmp = finish.reshape(HW, z)
                for coord in coords:
                    start_tmp[coord, :] = finish_tmp[coord, :]
                start = start_tmp.reshape(start.shape)

        scores = np.asarray(scores)
        return scores
