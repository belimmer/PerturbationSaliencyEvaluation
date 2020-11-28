'''
This module was adapted from a module in https://github.com/HuTobias/HIGHLIGHTS-LRP
Date: 2020
commit: 834bf795ee37a74b611beb79851438e9a8afd676
License: MIT

This module implements sanity checks for saliency maps.
To this end the layers in the model are cascadingly randomized and for each step we create a copy of the model.
Then we create gameplay and saliency map streams for each of those models, using the decisions of the original model,
 such that all models get the same input states.
Finally we compare the generated saliency of all models.
'''

from explanation import explainer

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
import pandas as pd
import keras
import seaborn as sns
import gym
from custom_atari_wrapper import atari_wrapper

import tensorflow as tf
import keras.backend as K


def show_and_save_plt(ax ,file_name, y_label=None, ylim =None, label_size = 18, tick_size = 14):
    """
    Shows and saves the given plot and defines the appearance of the final plot.
    :param ax: the plot to be saved.
    :param file_name: save file name where the file is saved.
    :param y_label: the y axis label displayed
    :param title: titel of displayed in the plot (currently not used)
    :param ylim: limits of the y axis.
    :param label_size: font size of the label text
    :param tick_size: font size of the tick numbers
    """
    #this only works the second time the function is used, since it sets the style for future plots.
    # It was still more convenient this way. #TODO fix this
    sns.set_style("whitegrid")

    if y_label != None:
        plt.ylabel(y_label)
    plt.xlabel(None)
    if ylim != None:
        ax.set(ylim=ylim)

    try:
        ax.yaxis.label.set_size(label_size)
        ax.xaxis.label.set_size(label_size)
    except:
        try:
            plt.ylabel(y_label, fontsize=label_size)
            plt.xlabel(fontsize=label_size)
        except Exception as e:
            print(e)

    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    file_name = os.path.join('figures/sanity_checks', file_name)
    if not (os.path.isdir(file_name)):
        os.makedirs(file_name)
        os.rmdir(file_name)
    plt.tight_layout()
    plt.savefig(file_name)

    plt.show()


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


def rand_layer(layer, mean = 0, SD = 0.1):
    '''Custom layer randomization for testing purposes.'''
    weights_shape = layer.get_weights()[0].shape
    bias_shape = layer.get_weights()[1].shape
    rand_weights =  np.random.normal(mean,SD,weights_shape)
    rand_bias = np.random.normal(mean,SD,bias_shape)
    layer.set_weights([rand_weights, rand_bias])


def init_layer(layer):
    ''' Re-initializes the given layer with the original initializer to achieve randomization of the layer that is
    within reasonable bounds for that layer.
    :param layer: the layer to be randomized
    :return: nothing, the given layer is randomized
    '''
    session = K.get_session()
    weights_initializer = tf.variables_initializer(layer.weights)
    session.run(weights_initializer)


def copy_model(model):
    '''
    Copies a keras model including the weights
    :param model: the model to be copied
    :return: the new copy of the model
    '''
    model_m1 = keras.models.clone_model(model)
    model_m1.set_weights(model.get_weights())
    return model_m1


def check_models(model1, model):
    ''' checks if two models have the same weights, to make sure that a layer was randomized.'''
    for i in range(1,7):
        if i != 4:
            print('layer ', i)
            print( (model1.get_layer(index=i).get_weights()[0] == model.get_layer(index=i).get_weights()[0]).all() )
            print( (model1.get_layer(index=i).get_weights()[1] == model.get_layer(index=i).get_weights()[1]).all() )


def calc_sim(learned_relevance, random_relevance, explainer):
    ''' Helper function to calculate the similarities of two saliency maps (for learned weights and partly random wheights).
    Only works in this code, since the similarity lists are created elsewhere. '''
    #normalizing:
    learned_relevance = normalise_image(learned_relevance)
    random_relevance = normalise_image(random_relevance)
    neg_random_relevance = 1 - random_relevance

    spearman, spearman2 = spearmanr(random_relevance.flatten(), learned_relevance.flatten(), nan_policy='omit')
    test, _ = spearmanr(neg_random_relevance.flatten(), learned_relevance.flatten(), nan_policy='omit')
    spearman = max(spearman, test)

    # ssim_val = ssim(random_relevance,learned_relevance, multichannel=True)
    ssim_val = ssim(random_relevance.flatten(), learned_relevance.flatten())
    test = ssim(neg_random_relevance.flatten(), learned_relevance.flatten())
    ssim_val = max(ssim_val, test)

    random_hog = hog(random_relevance)
    learned_hog = hog(learned_relevance)
    pearson, _ = pearsonr(random_hog, learned_hog)

    neg_random_hog = hog(neg_random_relevance)
    test, _ = pearsonr(neg_random_hog, learned_hog)
    pearson = max(pearson,test)

    pearson_list[explainer].append(abs(pearson))
    ssim_list[explainer].append(abs(ssim_val))
    spearman_list[explainer].append(abs(spearman))


if __name__ == '__main__':
    # if True, the agent plays a new game, otherwise an old stream is used
    # only needed for the first run
    create_stream = True
    # if True, the similarities are recalculated, otherwise old calculated similarities are loaded
    read_csv = True

    steps = 10

    directory = ''
    save_file_argmax_raw = os.path.join(directory, 'raw_argmax', 'raw_argmax')
    save_file_state = os.path.join(directory, 'state', 'state')
    save_file1 = os.path.join(directory, 'model1', 'raw_argmax')
    save_file2 = os.path.join(directory, 'model2', 'raw_argmax')
    save_file3 = os.path.join(directory, 'model3', 'raw_argmax')
    save_file4 = os.path.join(directory, 'model4', 'raw_argmax')
    save_file5 = os.path.join(directory, 'model5', 'raw_argmax')

    #create empty list to be filled later
    pearson_list = np.empty(5, dtype=np.object)
    pearson_list[:] = [], [], [], [], []
    ssim_list = np.empty(5, dtype=np.object)
    ssim_list[:] = [], [], [], [], []
    spearman_list = np.empty(5, dtype=np.object)
    spearman_list[:] = [], [], [], [], []
    model_list = []
    action_list = []

    if create_stream:
        # generate stream of states, actions, and saliency maps
        np.random.seed(42)

        model = keras.models.load_model('models/MsPacman_5M_ingame_reward.h5')

        model.summary()

        # create analyzer for fully trained model
        analyzer_arg = explainer(model)

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

        env = gym.make('MsPacmanNoFrameskip-v4')
        env.reset()
        wrapper = atari_wrapper(env)
        wrapper.reset(noop_max=1)
        env = gym.make('MsPacmanNoFrameskip-v4')
        env.reset()
        wrapper = atari_wrapper(env)
        wrapper.reset(noop_max=1)
        for _ in range(steps):
            if _ < 4:
                action = env.action_space.sample()
            else:
                my_input = np.expand_dims(stacked_frames, axis=0)
                output = model.predict(my_input)

                action = np.argmax(np.squeeze(output))

                original_occlusion_explanation = analyzer_arg.generate_occlusion_explanation(input=stacked_frames, use_softmax=True)
                original_rise_explanation = analyzer_arg.generate_rise_prediction(input=stacked_frames, use_softmax=True)
                original_noise_explanation = analyzer_arg.generate_greydanus_explanation(input=stacked_frames, blur=False)
                original_noise_blur_explanation = analyzer_arg.generate_greydanus_explanation(input=stacked_frames,
                                                                                         blur=True)
                original_lime_explanation = analyzer_arg.generate_lime_explanation(input=stacked_frames, rgb_image=False)[1]

                rand_occlusion_explanation1 = analyzer1.generate_occlusion_explanation(input=stacked_frames, use_softmax=True)
                rand_rise_explanation1 = analyzer1.generate_rise_prediction(input=stacked_frames, use_softmax=True)
                rand_noise_explanation1 = analyzer1.generate_greydanus_explanation(input=stacked_frames, blur=False)
                rand_noise_blur_explanation1 = analyzer1.generate_greydanus_explanation(input=stacked_frames, blur=True)
                rand_lime_explanation1 = analyzer1.generate_lime_explanation(input=stacked_frames, rgb_image=False)[1]
                calc_sim(original_occlusion_explanation, rand_occlusion_explanation1, 0)
                calc_sim(original_rise_explanation, rand_rise_explanation1, 1)
                calc_sim(original_noise_explanation, rand_noise_explanation1, 2)
                calc_sim(original_noise_blur_explanation, rand_noise_blur_explanation1, 3)
                calc_sim(original_lime_explanation, rand_lime_explanation1, 4)
                action_list.append(action)
                model_list.append(1)

                rand_occlusion_explanation2 = analyzer2.generate_occlusion_explanation(input=stacked_frames, use_softmax=True)
                rand_rise_explanation2 = analyzer2.generate_rise_prediction(input=stacked_frames, use_softmax=True)
                rand_noise_explanation2 = analyzer2.generate_greydanus_explanation(input=stacked_frames, blur=False)
                rand_noise_blur_explanation2 = analyzer2.generate_greydanus_explanation(input=stacked_frames, blur=True)
                rand_lime_explanation2 = analyzer2.generate_lime_explanation(input=stacked_frames, rgb_image=False)[1]
                calc_sim(original_occlusion_explanation, rand_occlusion_explanation2, 0)
                calc_sim(original_rise_explanation, rand_rise_explanation2, 1)
                calc_sim(original_noise_explanation, rand_noise_explanation2, 2)
                calc_sim(original_noise_blur_explanation, rand_noise_blur_explanation2, 3)
                calc_sim(original_lime_explanation, rand_lime_explanation2, 4)
                action_list.append(action)
                model_list.append(2)

                rand_occlusion_explanation3 = analyzer3.generate_occlusion_explanation(input=stacked_frames, use_softmax=True)
                rand_rise_explanation3 = analyzer3.generate_rise_prediction(input=stacked_frames, use_softmax=True)
                rand_noise_explanation3 = analyzer3.generate_greydanus_explanation(input=stacked_frames, blur=False)
                rand_noise_blur_explanation3 = analyzer3.generate_greydanus_explanation(input=stacked_frames, blur=True)
                rand_lime_explanation3 = analyzer3.generate_lime_explanation(input=stacked_frames, rgb_image=False)[1]
                calc_sim(original_occlusion_explanation, rand_occlusion_explanation3, 0)
                calc_sim(original_rise_explanation, rand_rise_explanation3, 1)
                calc_sim(original_noise_explanation, rand_noise_explanation3, 2)
                calc_sim(original_noise_blur_explanation, rand_noise_blur_explanation3, 3)
                calc_sim(original_lime_explanation, rand_lime_explanation3, 4)
                action_list.append(action)
                model_list.append(3)

                rand_occlusion_explanation4 = analyzer4.generate_occlusion_explanation(input=stacked_frames, use_softmax=True)
                rand_rise_explanation4 = analyzer4.generate_rise_prediction(input=stacked_frames, use_softmax=True)
                rand_noise_explanation4 = analyzer4.generate_greydanus_explanation(input=stacked_frames, blur=False)
                rand_noise_blur_explanation4 = analyzer4.generate_greydanus_explanation(input=stacked_frames, blur=True)
                rand_lime_explanation4 = analyzer4.generate_lime_explanation(input=stacked_frames, rgb_image=False)[1]
                calc_sim(original_occlusion_explanation, rand_occlusion_explanation4, 0)
                calc_sim(original_rise_explanation, rand_rise_explanation4, 1)
                calc_sim(original_noise_explanation, rand_noise_explanation4, 2)
                calc_sim(original_noise_blur_explanation, rand_noise_blur_explanation4, 3)
                calc_sim(original_lime_explanation, rand_lime_explanation4, 4)
                action_list.append(action)
                model_list.append(4)

                rand_occlusion_explanation5 = analyzer5.generate_occlusion_explanation(input=stacked_frames, use_softmax=True)
                rand_rise_explanation5 = analyzer5.generate_rise_prediction(input=stacked_frames, use_softmax=True)
                rand_noise_explanation5 = analyzer5.generate_greydanus_explanation(input=stacked_frames, blur=False)
                rand_noise_blur_explanation5 = analyzer5.generate_greydanus_explanation(input=stacked_frames, blur=True)
                rand_lime_explanation5 = analyzer5.generate_lime_explanation(input=stacked_frames, rgb_image=False)[1]
                calc_sim(original_occlusion_explanation, rand_occlusion_explanation5, 0)
                calc_sim(original_rise_explanation, rand_rise_explanation5, 1)
                calc_sim(original_noise_explanation, rand_noise_explanation5, 2)
                calc_sim(original_noise_blur_explanation, rand_noise_blur_explanation5, 3)
                calc_sim(original_lime_explanation, rand_lime_explanation5, 4)
                action_list.append(action)
                model_list.append(5)
            stacked_frames, observations, reward, done, info = wrapper.step(action)
            env.render()
            print(_)

        explainer_list = ['occlusion', 'rise', 'noise_black', 'noise_blur', 'lime']
        print("creating plots")
        for i in range(5):
            data_frame = pd.DataFrame(columns=['rand_layer', 'pearson', 'ssim', 'spearman', 'action'])

            data_frame['rand_layer'] = model_list
            data_frame['pearson'] = pearson_list[i]
            data_frame['ssim'] = ssim_list[i]
            data_frame['spearman'] = spearman_list[i]
            data_frame['action'] = action_list

            data_frame['rand_layer'] = data_frame.rand_layer.apply(
                lambda
                    x: 'fc_2' if x == 1 else 'fc_1' if x == 2 else 'conv_3' if x == 3 else 'conv_2' if x == 4 else 'conv_1')

            sns.set(palette='colorblind', style="whitegrid")

            ax = sns.barplot(x='rand_layer', y='pearson', data=data_frame)
            show_and_save_plt(ax, 'pearson_' + explainer_list[i], ylim=(0,1), label_size=28, tick_size=20, y_label='Pearson')
            ax = sns.barplot(x='rand_layer', y='ssim', data=data_frame)
            plt.xlabel(None)
            show_and_save_plt(ax, 'ssim_' + explainer_list[i], ylim=(0,1), label_size=28, tick_size=20, y_label='Ssim')
            ax = sns.barplot(x='rand_layer', y='spearman', data=data_frame)
            plt.xlabel(None)
            show_and_save_plt(ax, 'spearman_' + explainer_list[i], ylim=(0,1), label_size=28, tick_size=20, y_label='Spearman')