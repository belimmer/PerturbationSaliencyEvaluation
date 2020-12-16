import keras
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import re
import timeit

from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

import applications.atari.main_atari as main_atari
import applications.atari.rise as rise
import applications.atari.greydanus as greydanus
import applications.atari.custom_occlusion_sensitvity as custom_occlusion_sensitvity
import applications.atari.custom_greydanus as custom_greydanus
import applications.atari.custom_lime as custom_lime

import gym
from applications.atari.custom_atari_wrapper import atari_wrapper

class explainer():
    """
    Class to generate different occlusion based explanations
    """
    def __init__(self, model):
        self.model = model
        self.rise_masks = []
        self.masks_generated = False

    def generate_occlusion_explanation(self, input, patch_size=5, use_softmax=False, use_old_confidence=False, color=0.0):
        """
        Generates an explanation using the Occlusion Sensitivity approach.

        Args:
            input: image which will be explained
            patch_size (int): size of the square used to occlude the image
            use_softmax (bool): should a softmax be used for the output of the model

        Returns:
            saliency_map: a saliency map which functions as explanation
        """
        probabilities = np.squeeze(self.model.predict(np.expand_dims(input, axis=0)))
        proposed_action = np.argmax(probabilities)
        explainer = custom_occlusion_sensitvity.CustomOcclusionSensitivity()
        saliency_map = explainer.get_sensitivity_map(image=input, model=self.model, class_index=proposed_action,
                                                     patch_size=patch_size, use_softmax=use_softmax, use_old_confidence=use_old_confidence, color=color)
        return saliency_map

    def generate_lime_explanation(self, rgb_image, input, hide_img=True, positive_only=False, num_features=3):
        """
        Generates an explanation using the LIME approach.

        Args:
            rgb_image (bool): is the image an RGB image (relevant for the segmentation algorithm)
            input: image which will be explained
            hide_img (bool): should the parts of the image not relevant to the explanation be greyed out
            positive_only (bool): should only parts of the image which positively impact the prediction be highlighted

        Returns:
            stacked_explanation: explanation produced by LIME
            mask: shows the most important super pixels
            ranked_mask: shows the most important super pixels and ranks them by importance
        """
        random_seed = np.random.randint(0, high=1000)
        segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=2,
                                                max_dist=10, ratio=0.3,
                                                random_seed=random_seed, convert2lab=rgb_image)

        lime_explainer = custom_lime.CustomLimeImageExplainer()
        explanation = lime_explainer.custom_explain_instance(input, self.model.predict, segmentation_fn=segmentation_fn,
                                                      top_labels=1, hide_color=0,
                                                      num_samples=1000)
        stacked_explanation, mask, ranked_mask = explanation.custom_get_image_and_mask(explanation.top_labels[0],
                                                                                       positive_only=positive_only,
                                                                                       num_features=num_features,
                                                                                       hide_rest=hide_img)
        return stacked_explanation, mask, ranked_mask

    def generate_greydanus_explanation(self, input, r=5, blur=True):
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
        return explainer.generate_explanation(input, self.model, radius=r)

    def generate_rise_prediction(self, input, probability=0.9, use_softmax=True):
        """
        Generates an explanation using the LIME approach.

        Args:
            input: image which will be explained
            probability: probability for a mask to blur a pixel of the image

        Returns:
            a saliency map which functions as explanation
        """
        N = 2000  #number of masks
        s = 8
        p1 = probability  #probability to not occlude a pixel
        batch_size = 1
        input_size = (input.shape[0], input.shape[1])
        explainer = rise.rise_explainer(N, s, p1, batch_size)
        if not self.masks_generated:
            self.rise_masks = explainer.generate_masks(input_size)
            self.masks_generated = True
        prediction = explainer.explain(self.model, np.expand_dims(input, axis=0), self.rise_masks, input_size, use_softmax=use_softmax)
        model_prediction = self.model.predict(np.expand_dims(input, axis=0))
        model_prediction = np.argmax(np.squeeze(model_prediction))
        return prediction[model_prediction]


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
    main_atari.save_frame(saliency, output_path + "_saliency_only.png", 0)

    plt.axis('off')
    plt.imshow(image)
    plt.imshow(saliency, cmap=cmap, alpha=0.6)
    plt.savefig(output_path + ".png")


def create_lime_image(mask, frames, output_path, shape):
    """
    Function to visualize LIME explanations.

    Args:
        mask: mask produced by LIME
        frames: frames which will be modified
        output_path: path were the modified frames are saved
        shape: shape of the frames
    """
    custom_mask = Image.fromarray(mask.astype(np.float32)).resize(size=shape, resample=Image.NEAREST)
    custom_mask = np.array(custom_mask)

    for i, frame in enumerate(frames):
        custom_mask.astype("uint8")
        copied_frame = frame.copy()
        for y in range(custom_mask.shape[0]):
            for x in range(custom_mask.shape[1]):
                if custom_mask[y][x] == 1:
                    copied_frame[y][x][1] = 170
        original_with_boundaries = mark_boundaries(copied_frame, custom_mask.astype("uint8"))
        main_atari.save_frame(original_with_boundaries, output_path, i)


def insertion_for_all(_, my_explainer, stacked_frames, model, plot = False):
    """
    calculate the insertion metric for all saliency map approaches for the input *stacked_frames*
    :param _: the frame number, used for naming single plots
    :param my_explainer: the explainer used to create saliency maps
    :param stacked_frames: the input frames for the agent
    :param model: the DQN agent
    :param plot: set this true to plot intermediate results
    :return: a list containing the insertion metric scores for each sliency map approach.
    """
    tmp_scores = []

    # Occlusion Sensitivity
    saliency_map = my_explainer.generate_occlusion_explanation(input=np.squeeze(stacked_frames),
                                                               use_softmax=True)
    insertion = rise.CausalMetric(model=model, mode='ins', step=np.squeeze(stacked_frames).shape[0],
                                  substrate_fn=rise.custom_black)
    score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map,
                                 name="frame_" + str(_), approach="occl", plot=plot, use_softmax=True)
    tmp_scores.append(score)

    # Occlusion Sensitivity grey
    saliency_map = my_explainer.generate_occlusion_explanation(input=np.squeeze(stacked_frames),
                                                               use_softmax=True, color=0.5)
    score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map,
                                 name="frame_" + str(_), approach="occl", plot=False, use_softmax=True)
    tmp_scores.append(score)

    # Noise Sensitivity with black occlusion
    saliency_map = my_explainer.generate_greydanus_explanation(input=np.squeeze(stacked_frames), blur=False)
    score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map,
                                 name="frame_" + str(_), approach="noise", plot=plot, use_softmax=True)
    tmp_scores.append(score)

    # Noise Sensitivity with gaussian noise
    saliency_map = my_explainer.generate_greydanus_explanation(input=np.squeeze(stacked_frames), blur=True)
    score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map,
                                 name="frame_" + str(_), approach="noise_blur", plot=plot, use_softmax=True)
    tmp_scores.append(score)

    # LIME
    explanation, mask, ranked_mask = my_explainer.generate_lime_explanation(rgb_image=False,
                                                                            input=np.squeeze(
                                                                                stacked_frames),
                                                                            hide_img=False,
                                                                            positive_only=False)
    score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=ranked_mask,
                                 name="frame_" + str(_), approach="lime", plot=plot, use_softmax=True)
    tmp_scores.append(score)

    # RISE
    saliency_map = my_explainer.generate_rise_prediction(input=np.squeeze(stacked_frames))
    score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map,
                                 name="frame_" + str(_),
                                 approach="rise", plot=plot, use_softmax=True)
    tmp_scores.append(score)

    # Random
    saliency_map = np.random.random(size=saliency_map.shape)
    score = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map,
                                 name="frame_" + str(_), approach="noise_blur", plot=plot, use_softmax=True)
    tmp_scores.append(score)

    return tmp_scores

if __name__ == '__main__':
    state_path = "HIGHLIGHTS_states/"
    state_output_path = "output_highlight_states/"
    model = keras.models.load_model('models/MsPacman_5M_ingame_reward.h5')
    my_explainer = explainer(model=model)

    states = []
    state_numbers = []
    occlusion_time = []
    lime_time = []
    greydanus_time = []
    rise_time = []
    occlusion_auc = []
    occlusion_old_confidence_auc = []
    lime_auc = []
    greydanus_auc = []
    rise_auc = []
    greydanus_auc_noise = []
    simulate_game = True  # take the states from highlight states or simulate the game

    if simulate_game:
        # use a different start to introduce randomnes
        fixed_start = True

        # for reward calculation
        total_reward = 0
        reward_list = []
        scores = []

        np.random.seed(42)

        steps = 1001
        env = gym.make('MsPacmanNoFrameskip-v4')
        env.reset()
        wrapper = atari_wrapper(env)
        wrapper.reset(noop_max=1)
        plot = False
        if fixed_start:
            wrapper.fixed_reset(1, 0)  # used  action 3 and 4

        for _ in range(steps):
            if _ < 4:
                action = env.action_space.sample()
                # to have more controll over the fixed starts
                if fixed_start:
                    action = 0
            else:
                my_input = np.expand_dims(stacked_frames, axis=0)
                output = model.predict(
                    my_input)  # this output corresponds with the output in baseline if --dueling=False is correctly set for baselines.
                # save model predictions

                action = np.argmax(np.squeeze(output))

                if _ % 10 == 0:
                    print("Step: " + str(_))
                if _ % 50 == 0:
                    plot = True
                else:
                    plot = False

                tmp_scores = insertion_for_all(_, my_explainer, stacked_frames, model, plot = False)

                scores.append(tmp_scores)

            stacked_frames, observations, reward, done, info = wrapper.step(action)

            total_reward += reward
            if done:
                print('total_reward', total_reward)
                reward_list.append(total_reward)
                total_reward = 0
            env.render()

            if _ % 100 == 0 and _ != 0:
                print("Saving progress...")
                np.save(file="figures/backup_predictions/pred_" + str(_), arr=scores)
                scores = []

        reward_list.append(total_reward)
        average_reward = np.mean(reward_list)
        print(average_reward)
    else:
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

            start = timeit.default_timer()
            saliency_map = my_explainer.generate_occlusion_explanation(input=np.squeeze(state[0]), use_softmax=True)
            stop = timeit.default_timer()
            occlusion_time.append(stop - start)
            create_saliency_image(saliency_map=saliency_map, image=original_frames[3],
                                  output_path=state_output_path + "occlusion_explanation_" + state[1], cmap="viridis")
            insertion = rise.CausalMetric(model=model, mode='ins', step=np.squeeze(state[0]).shape[0],
                                          substrate_fn=rise.custom_black)
            score = insertion.single_run(img_tensor=np.squeeze(state[0]), explanation=saliency_map, name=state[1],
                                         approach="occl", use_softmax=True)
            print("Occlusion Sensitivity: " + state[1] + " AUC: " + str(rise.auc(score)))

            start = timeit.default_timer()
            saliency_map = my_explainer.generate_greydanus_explanation(input=np.squeeze(state[0]), blur=False)
            stop = timeit.default_timer()
            greydanus_time.append(stop - start)
            create_saliency_image(saliency_map=saliency_map, image=original_frames[3],
                                  output_path=state_output_path + "greydanus_explanation_" + state[1])
            score = insertion.single_run(img_tensor=np.squeeze(state[0]), explanation=saliency_map, name=state[1],
                                         approach="noise", use_softmax=True)
            print("Noise Sensitivity: " + state[1] + " AUC: " + str(rise.auc(score)))

            start = timeit.default_timer()
            explanation, mask, ranked_mask = my_explainer.generate_lime_explanation(rgb_image=False,
                                                                                    input=np.squeeze(state[0]),
                                                                                    hide_img=False, positive_only=False)
            stop = timeit.default_timer()
            lime_time.append(stop - start)
            create_lime_image(mask=mask, frames=original_frames,
                              output_path=state_output_path + "lime_explanation_" + state[1], shape=(160, 210))

            explanation, mask, ranked_mask = my_explainer.generate_lime_explanation(rgb_image=False,
                                                                                    input=np.squeeze(state[0]),
                                                                                    hide_img=False, positive_only=True)
            score = insertion.single_run(img_tensor=np.squeeze(state[0]), explanation=ranked_mask, name=state[1],
                                         approach="lime", use_softmax=True)
            print("LIME: " + state[1] + " AUC: " + str(rise.auc(score)))

            start = timeit.default_timer()
            saliency_map = my_explainer.generate_rise_prediction(input=np.squeeze(state[0]))
            stop = timeit.default_timer()
            rise_time.append((stop - start) / 2)
            create_saliency_image(saliency_map=saliency_map, image=original_frames[3],
                                  output_path=state_output_path + "rise_explanation_" + state[1])
            score = insertion.single_run(img_tensor=np.squeeze(state[0]), explanation=saliency_map, name=state[1],
                                         approach="rise", use_softmax=True)
            print("RISE: " + state[1] + " AUC: " + str(rise.auc(score)))

    if not simulate_game:
        print("Occlusion average Time: ")
        print(sum(occlusion_time) / len(occlusion_time))
        print("LIME average Time:")
        print(sum(lime_time) / len(lime_time))
        print("Greydanus average Time:")
        print(sum(greydanus_time) / len(greydanus_time))
        print("RISE average Time:")
        print(sum(rise_time) / len(greydanus_time))