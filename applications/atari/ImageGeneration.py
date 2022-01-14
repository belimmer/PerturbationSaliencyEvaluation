"""
Module for creating example saliency map images for the HIGHLIGHT states.

Some functions are adapted from https://github.com/HuTobias/HIGHLIGHTS-LRP
Date: 2020
commit: 834bf795ee37a74b611beb79851438e9a8afd676
License: MIT
"""

import keras
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import re

from applications.atari.explanation import save_frame

import os
from skimage import filters, transform
import skimage
from applications.atari.sanity_checks.sanity_checks_main import init_layer, copy_model, check_models

from applications.atari.explanation import explainer
import skimage.segmentation as seg
import applications.atari.used_parameters as used_parameters

import timeit

def save_image(file_name, image, **kwargs):
    """
    saves image under file_name and creates the directory if it does not exist yet
    :param file_name:
    :param image:
    :return: nothing
    """
    if not (os.path.isdir(file_name)):
        os.makedirs(file_name)
        os.rmdir(file_name)
    # TODO added cmap
    plt.imsave(file_name, image, **kwargs)


def add_saliency_to_image(saliency, image, saliency_brightness=1, saliency_power=2):
    '''
    adds a saliency map(in green) over a given image
    :param saliency: the saliency map to be applied
    :param image: the original image
    :param saliency_brightness: the brightness of the saliency map
    :return: the overlayed image
    '''

    image_shape = (image.shape[0], image.shape[1])
    saliency = transform.resize(saliency, image_shape, order=0, mode='reflect')
    zeros = np.zeros(image_shape)
    saliency = np.power(saliency, saliency_power)
    saliency = np.stack((zeros, saliency, zeros), axis=-1)
    saliency *= saliency_brightness
    final_image = image + saliency
    final_image = np.clip(final_image, 0, 1)
    return final_image


def create_edge_image(image):
    ''' creates a edge version of an image
    :param image: the original image
    :return: edge only version of the image
    '''
    image = skimage.color.rgb2gray(image)
    image = filters.sobel(image)
    image = np.stack((image, image, image), axis=-1)
    return image


def output_saliency_map(saliency, image, scale_factor=3, saliency_factor=2, edges=True):
    ''' scales the image and adds the saliency map
    :param saliency:
    :param image:
    :param scale_factor: factor to scale height and width of the image
    :param saliency_factor:
    :param edges: if True, creates a edge version of the image first
    :return:
    '''
    image = np.squeeze(image)
    output_shape = (image.shape[0] * scale_factor, image.shape[1] * scale_factor)
    image = transform.resize(image, output_shape, order=0, mode='reflect')
    if edges:
        image = create_edge_image(image, output_shape)

    final_image = add_saliency_to_image(saliency, image, saliency_factor)

    return final_image


def normalise_image(image):
    '''normalises image by forcing the min and max values to 0 and 1 respectively
     :param image: the input image
    :return: normalised image as numpy array
    '''
    try:
        image = np.asarray(image)
    except:
        print('Cannot convert image to array')
    image = image - image.min()
    if image.max() != 0:
        image = image / image.max()
    return image


def create_saliency_image(saliency_map, image, output_path, cmap='jet'):
    """
    Takes an image and ads a saliency map as color map.

    Args:
        saliency_map: saliency map which will be used as color map
        image: image which is the saliency map is added onto
        output_path: path were the modified image is saved
        cmap: type of color map
    """

    image_shape = (336, 336)
    saliency = transform.resize(saliency_map, image_shape, order=0, mode='reflect')
    image = transform.resize(image, image_shape, order=0, mode='reflect')

    saliency = np.array(saliency)
    save_frame(saliency, output_path + "_saliency_only", 0)


    saliency = normalise_image(saliency)
    new_img = add_saliency_to_image(saliency, image, saliency_power=2)
    save_image(output_path + ".png", new_img)
    image = create_edge_image(image)
    new_img = add_saliency_to_image(saliency, image, saliency_power=2, saliency_brightness=0.8)
    save_image(output_path + "_edge.png", new_img)

if __name__ == '__main__':
    state_path_base = "HIGHLIGHTS_states/"
    state_output_path_base = "output_highlight_states/"

    # games = ["pacman"]
    games = ["pacman", "frostbite", "SpaceInvaders", "Breakout"]

    for GAME in games:
        if GAME == "pacman":
            state_path = state_path_base
            state_output_path = state_output_path_base
            model = keras.models.load_model('models/MsPacman_5M_ingame_reward.h5')
        if GAME == "frostbite":
            state_path = os.path.join(state_path_base, GAME) + "/"
            state_output_path = os.path.join(state_output_path_base, GAME) + "/"
            model = keras.models.load_model('models/FrostbiteIngame_5M.h5')
        if GAME == "SpaceInvaders":
            state_path = os.path.join(state_path_base, GAME) + "/"
            state_output_path = os.path.join(state_output_path_base, GAME) + "/"
            model = keras.models.load_model('models/SpaceInvadersIngame_5M.h5')
        if GAME == "Breakout":
            state_path = os.path.join(state_path_base, GAME) + "/"
            state_output_path = os.path.join(state_output_path_base, GAME) + "/"
            model = keras.models.load_model('models/BreakoutIngame_5M.h5')

        approach = "lime"
        segmentation = "quickshift"

        if approach == "rise":
            kwargs = {"probability" : used_parameters.RISE_PROBABILITY, "mask_size" : used_parameters.RISE_MASK_SIZE,
                      "number_of_mask" : used_parameters.RISE_NUM_MASKS, "use_softmax": used_parameters.RISE_SOFTMAX}
        if approach == "occl":
            kwargs = {"patch_size": used_parameters.OCCL_PATCH_SIZE, "color": used_parameters.OCCL_COLOR,
                      "use_softmax": used_parameters.OCCL_SOFTMAX}
        if approach == "noise":
            BLUR = True
            RAW_DIFF = False
            kwargs = {"r": used_parameters.NOISE_RADIUS, "blur":BLUR, "raw_diff": RAW_DIFF}
        if approach == "sarfa":
            BLUR = True
            kwargs = {"r": used_parameters.SARFA_RADIUS, "blur": used_parameters.SARFA_BLUR}
        if approach == "lime":
            kwargs = {"hide_img": False, "positive_only": True, "num_features": 5}
            if segmentation == "slic":
                segmentation_fn = (lambda x: seg.slic(x, n_segments=used_parameters.SLIC_N_SEGMENTS,
                                                      compactness=used_parameters.SLIC_COMPACTNESS,
                                                      sigma=used_parameters.SLIC_SIGMA))
                kwargs["segmentation_fn"] = segmentation_fn
                kwargs["num_samples"] = used_parameters.SLIC_NUM_SAMPLES
            elif segmentation == "quickshift":
                segmentation_fn = (lambda x: seg.quickshift(x, kernel_size=used_parameters.QUICKSHIFT_KERNEL_SIZE,
                                                            max_dist=used_parameters.QUICKSHIFT_MAX_DIST,
                                                            ratio=used_parameters.QUICKSHIFT_RATIO, convert2lab=False))
                kwargs["segmentation_fn"] = segmentation_fn
                kwargs["num_samples"] = used_parameters.QUICKSHIFT_NUM_SAMPLES
            elif segmentation == "felzenswalb":
                segmentation_fn = ( lambda x: seg.felzenszwalb(x, scale=used_parameters.FELZ_SCALE,
                                                               sigma=used_parameters.FELZ_SIGMA,
                                                               min_size=used_parameters.FELZ_MIN_SIZE))
                kwargs["segmentation_fn"] = segmentation_fn
                kwargs["num_samples"] = used_parameters.FELZ_NUM_SAMPLES
                # kwargs["num_features"] = 10

        save_dir = os.path.join(state_output_path, approach)
        if approach == "lime":
            save_dir = os.path.join(state_output_path, f"{approach}_{segmentation}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        states = []
        state_numbers = []

        # create analyzer for fully trained model
        original_analyzer = explainer(model)

        # create analyzer for model with randomized last layer
        model1 = copy_model(model)
        layer = model1.get_layer(index=6)
        init_layer(layer)
        check_models(model1, model)
        analyzer1 = explainer(model1)

        # create analyzer for model where the two last layers are randomized
        model2 = copy_model(model1)
        layer = model2.get_layer(index=5)
        init_layer(layer)
        check_models(model2, model1)
        analyzer2 = explainer(model2)

        # create analyzer for model where the three last layers are randomized
        model3 = copy_model(model2)
        layer = model3.get_layer(index=3)
        init_layer(layer)
        check_models(model3, model2)
        analyzer3 = explainer(model3)

        # create analyzer for model where the four last layers are randomized
        model4 = copy_model(model3)
        layer = model4.get_layer(index=2)
        init_layer(layer)
        check_models(model4, model3)
        analyzer4 = explainer(model4)

        # create analyzer for model where all layers are randomized
        model5 = copy_model(model4)
        layer = model5.get_layer(index=1)
        init_layer(layer)
        check_models(model5, model4)
        analyzer5 = explainer(model5)

        if approach == "noise":
            og_saliency_fn = (
                lambda kwargs: original_analyzer.generate_greydanus_explanation(**kwargs))
            saliency_fn_1 = (lambda kwargs: analyzer1.generate_greydanus_explanation(**kwargs))
            saliency_fn_2 = (
                lambda kwargs: analyzer2.generate_greydanus_explanation(**kwargs))
            saliency_fn_3 = (
                lambda kwargs: analyzer3.generate_greydanus_explanation(**kwargs))
            saliency_fn_4 = (
                lambda kwargs: analyzer4.generate_greydanus_explanation(**kwargs))
            saliency_fn_5 = (
                lambda kwargs: analyzer5.generate_greydanus_explanation(**kwargs))
        #
        if approach == "occl":
            og_saliency_fn = (
                lambda kwargs: original_analyzer.generate_occlusion_explanation(**kwargs))
            saliency_fn_1 = (lambda kwargs: analyzer1.generate_occlusion_explanation(**kwargs))
            saliency_fn_2 = (
                lambda kwargs: analyzer2.generate_occlusion_explanation(**kwargs))
            saliency_fn_3 = (
                lambda kwargs: analyzer3.generate_occlusion_explanation(**kwargs))
            saliency_fn_4 = (
                lambda kwargs: analyzer4.generate_occlusion_explanation(**kwargs))
            saliency_fn_5 = (
                lambda kwargs: analyzer5.generate_occlusion_explanation(**kwargs))

        if approach == "rise":
            og_saliency_fn = (
                lambda kwargs: original_analyzer.generate_rise_prediction(**kwargs))
            saliency_fn_1 = (
                lambda kwargs: analyzer1.generate_rise_prediction(**kwargs))
            saliency_fn_2 = (
                lambda kwargs: analyzer2.generate_rise_prediction(**kwargs))
            saliency_fn_3 = (
                lambda kwargs: analyzer3.generate_rise_prediction(**kwargs))
            saliency_fn_4 = (
                lambda kwargs: analyzer4.generate_rise_prediction(**kwargs))
            saliency_fn_5 = (
                lambda kwargs: analyzer5.generate_rise_prediction(**kwargs))

        if approach == "lime":
            og_saliency_fn = (
                lambda kwargs: original_analyzer.generate_lime_explanation(**kwargs)[2])
            saliency_fn_1 = (
                lambda kwargs: analyzer1.generate_lime_explanation(**kwargs)[2])
            saliency_fn_2 = (
                lambda kwargs: analyzer2.generate_lime_explanation(**kwargs)[2])
            saliency_fn_3 = (
                lambda kwargs: analyzer3.generate_lime_explanation(**kwargs)[2])
            saliency_fn_4 = (
                lambda kwargs: analyzer4.generate_lime_explanation(**kwargs)[2])
            saliency_fn_5 = (
                lambda kwargs: analyzer5.generate_lime_explanation(**kwargs)[2])

        if approach == "sarfa":
            og_saliency_fn = (
                lambda kwargs: original_analyzer.generate_sarfa_explanation(**kwargs))
            saliency_fn_1 = (lambda kwargs: analyzer1.generate_sarfa_explanation(**kwargs))
            saliency_fn_2 = (
                lambda kwargs: analyzer2.generate_sarfa_explanation(**kwargs))
            saliency_fn_3 = (
                lambda kwargs: analyzer3.generate_sarfa_explanation(**kwargs))
            saliency_fn_4 = (
                lambda kwargs: analyzer4.generate_sarfa_explanation(**kwargs))
            saliency_fn_5 = (
                lambda kwargs: analyzer5.generate_sarfa_explanation(**kwargs))

        for state_name in os.listdir(path=state_path):
            if state_name.endswith(".npy"):
                states.append((np.load(state_path + state_name), re.findall("(\d+)", state_name)[0]))

        for state in states:
            original_frames = []
            for i in range(4):
                frame = Image.open(state_path + "screen_" + state[1] + "_" + str(i) + ".png")
                frame_array = np.array(frame)
                frame_array = frame_array[..., :3]
                original_frames.append(frame_array)

            plt.imshow(np.squeeze(state[0]))
            plt.show()

            last_frame = state[0][0,: , :, 3]
            last_frame = skimage.color.gray2rgb(last_frame)
            plt.imshow(last_frame)
            plt.show()

            if approach == "input_images":
                image = original_frames[3]
                image_shape = (336, 336)
                image = transform.resize(image, image_shape, order=0, mode='reflect')

                file_name = state[1]
                file_name = os.path.join(save_dir, file_name)
                save_image(file_name + ".png", image)
                continue


            kwargs["input"] = np.squeeze(state[0])
            action = np.argmax(model.predict(state[0]))
            kwargs["neuron_selection"] = action

            start = timeit.default_timer()
            saliency_map = og_saliency_fn(kwargs)
            stop = timeit.default_timer()
            time = stop - start
            print(time)

            test2 = normalise_image(saliency_map)
            plt.imshow(test2)
            plt.show()

            file_name = state[1] + "_explanation"
            file_name = os.path.join(save_dir, file_name)
            create_saliency_image(saliency_map=saliency_map, image=original_frames[3],
                                   output_path=file_name, cmap="viridis")

            # # set the rise masks such that the results do not depend on random maks generation
            if approach == "rise":
                analyzer1.rise_masks = original_analyzer.rise_masks
                analyzer1.masks_generated = True
                analyzer2.rise_masks = original_analyzer.rise_masks
                analyzer2.masks_generated = True
                analyzer3.rise_masks = original_analyzer.rise_masks
                analyzer3.masks_generated = True
                analyzer4.rise_masks = original_analyzer.rise_masks
                analyzer4.masks_generated = True
                analyzer5.rise_masks = original_analyzer.rise_masks
                analyzer5.masks_generated = True


            sanity_check_dir = os.path.join(save_dir, "sanit_checks")
            if not os.path.isdir(sanity_check_dir):
                os.makedirs(sanity_check_dir)

            saliency_map = saliency_fn_1(kwargs)
            file_name = state[1] + "_sanity_1"
            file_name = os.path.join(sanity_check_dir, file_name)
            create_saliency_image(saliency_map=saliency_map, image=original_frames[3],
                                  output_path=file_name, cmap="viridis")

            saliency_map = saliency_fn_2(kwargs)
            file_name = state[1] + "_sanity_2"
            file_name = os.path.join(sanity_check_dir, file_name)
            create_saliency_image(saliency_map=saliency_map, image=original_frames[3],
                                  output_path=file_name, cmap="viridis")

            saliency_map = saliency_fn_3(kwargs)
            file_name = state[1] + "_sanity_3"
            file_name = os.path.join(sanity_check_dir, file_name)
            create_saliency_image(saliency_map=saliency_map, image=original_frames[3],
                                  output_path=file_name, cmap="viridis")

            saliency_map = saliency_fn_4(kwargs)
            file_name = state[1] + "_sanity_4"
            file_name = os.path.join(sanity_check_dir, file_name)
            create_saliency_image(saliency_map=saliency_map, image=original_frames[3],
                                  output_path=file_name, cmap="viridis")

            saliency_map = saliency_fn_5(kwargs)
            file_name = state[1] + "_sanity_5"
            file_name = os.path.join(sanity_check_dir, file_name)
            create_saliency_image(saliency_map=saliency_map, image=original_frames[3],
                                  output_path=file_name, cmap="viridis")
