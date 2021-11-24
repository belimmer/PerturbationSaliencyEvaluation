import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import applications.atari.rise as rise
import applications.atari.greydanus as greydanus
import applications.atari.custom_occlusion_sensitvity as custom_occlusion_sensitvity
import applications.atari.custom_greydanus as custom_greydanus
import applications.atari.custom_lime as custom_lime
import applications.atari.sarfa as sarfa

import os


class explainer():
    """
    Class to generate different occlusion based explanations
    """
    def __init__(self, model):
        self.model = model
        self.rise_masks = []
        self.masks_generated = False

    def generate_occlusion_explanation(self, input, patch_size=5, use_softmax=False, use_old_confidence=False,
                                       color=0.0, neuron_selection = False, **kwargs):
        """
        Generates an explanation using the Occlusion Sensitivity approach.

        Args:
            input: image which will be explained
            patch_size (int): size of the square used to occlude the image
            use_softmax (bool): should a softmax be used for the output of the model
            neuron_selection (int): the index of the action that should be analyzed. Takes the highest action if False

        Returns:
            saliency_map: a saliency map which functions as explanation
        """
        if neuron_selection is False:
            probabilities = np.squeeze(self.model.predict(np.expand_dims(input, axis=0)))
            proposed_action = np.argmax(probabilities)
        else:
            proposed_action = neuron_selection
        explainer = custom_occlusion_sensitvity.CustomOcclusionSensitivity()
        saliency_map = explainer.get_sensitivity_map(image=input, model=self.model, class_index=proposed_action,
                                                     patch_size=patch_size, use_softmax=use_softmax,
                                                     use_old_confidence=use_old_confidence, color=color, **kwargs)
        return saliency_map

    def generate_lime_explanation(self, input, hide_img=True, positive_only=False, num_features=3, segmentation_fn=None
                                  , neuron_selection=False, num_samples = 1000):
        """
        Generates an explanation using the LIME approach.

        Args:
            input: image which will be explained
            hide_img (bool): should the parts of the image not relevant to the explanation be greyed out
            positive_only (bool): should only parts of the image which positively impact the prediction be highlighted
            segmentation_fn: the segmentation function used for the LIME explanation

        Returns:
            stacked_explanation: explanation produced by LIME
            mask: shows the most important super pixels
            ranked_mask: shows the most important super pixels and ranks them by importance
        """

        lime_explainer = custom_lime.CustomLimeImageExplainer()
        if neuron_selection is not False:
            top_labels = False
            labels = [neuron_selection]
        else:
            top_labels = 1
            labels = (1,)
        explanation = lime_explainer.custom_explain_instance(input, self.model.predict, segmentation_fn=segmentation_fn,
                                                      top_labels=top_labels, hide_color=0,
                                                      num_samples= num_samples, labels= labels)
        if neuron_selection is False:
            stacked_explanation, mask, ranked_mask = explanation.custom_get_image_and_mask(explanation.top_labels[0],
                                                                                       positive_only=positive_only,
                                                                                       num_features=num_features,
                                                                                       hide_rest=hide_img)
        else:
            stacked_explanation, mask, ranked_mask = explanation.custom_get_image_and_mask(neuron_selection,
                                                                                           positive_only=positive_only,
                                                                                           num_features=num_features,
                                                                                           hide_rest=hide_img)
        return stacked_explanation, mask, ranked_mask

    def generate_greydanus_explanation(self, input, r=5, blur=True, **kwargs):
        """
        Generates an explanation using the Noise Sensitivity (Greydanus) approach.

        Args:
            input: image which will be explained
            r (int): radius of the blur
            blur (bool): indicates if a gaussian blur should be used to occlude the image

        Returns:
            a saliency map which functions as explanation
        """
        if blur:
            explainer = greydanus.greydanus_explainer()
        else:
            explainer = custom_greydanus.custom_greydanus_explainer()
        return explainer.generate_explanation(input, self.model, radius=r, **kwargs)

    def generate_sarfa_explanation(self, input, r=5, blur=True, **kwargs):
        """
        Generates an explanation using the SARFA approach.

        Args:
            input: state which will be explained
            r (int): radius of the perturbation
            blur (bool): indicates if a gaussian blur should be used to occlude the image

        Returns:
            a saliency map which functions as explanation
        """
        sarfa_explainer = sarfa.SarfaExplainer()
        return sarfa_explainer.generate_explanation(input, self.model, radius=r, blur=blur, **kwargs)

    def generate_rise_prediction(self, input, probability=0.9, use_softmax = True, number_of_mask = 2000, mask_size=8
                                 , neuron_selection = False):
        """
        Generates an explanation using the RISE approach.

        Args:
            input: image which will be explained
            probability: probability for a mask to blur a pixel of the image
            use_softmax: should the softmax of the prediction be used for comparing different inputs
            number_of_mask: the number of calculated masks
            mask_size: the downscaled masks have size (mask_size x mask_size)
            neuron_selection: if this is not False, it gives the index of the action that should be explained

        Returns:
            a saliency map which functions as explanation
        """
        N = number_of_mask  #number of masks
        s = mask_size
        p1 = probability  #probability to not occlude a pixel
        batch_size = 1
        input_size = (input.shape[0], input.shape[1])
        explainer = rise.rise_explainer(N, s, p1, batch_size)
        if not self.masks_generated:
            self.rise_masks = explainer.generate_masks(input_size)
            self.masks_generated = True
        prediction = explainer.explain(self.model, np.expand_dims(input, axis=0), self.rise_masks,
                                       input_size, use_softmax = use_softmax)
        if neuron_selection is False:
            model_prediction = self.model.predict(np.expand_dims(input, axis=0))
            model_prediction = np.argmax(np.squeeze(model_prediction))
        else:
            model_prediction = neuron_selection
        return prediction[model_prediction]


def save_frame(array, save_file, frame):
    if not (os.path.isdir(save_file)):
        os.makedirs(save_file)
        os.rmdir(save_file)
    plt.imsave(save_file + '_' + str(frame) + '.png', array)


def create_saliency_image(saliency_map, image, output_path, cmap='jet'):
    """
    Takes an image and ads a saliency map as color map.

    Args:
        saliency_map: saliency map which will be used as color map
        image: image which is the saliency map is added onto
        output_path: path were the modified image is saved
        cmap: type of color map
    """
    saliency = Image.fromarray(saliency_map.astype(np.float32)).resize(size=(image.shape[1], image.shape[0]),
                                                                                       resample=Image.NEAREST)
    saliency = np.array(saliency)
    save_frame(saliency, output_path + "_saliency_only.png", 0)

    plt.axis('off')
    plt.imshow(image)
    plt.imshow(saliency, cmap=cmap, alpha=0.6)
    plt.savefig(output_path + ".png")
