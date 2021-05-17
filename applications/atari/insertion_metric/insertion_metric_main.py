from applications.atari.custom_atari_wrapper import atari_wrapper
from applications.atari.explanation import explainer
import gym
import keras
import numpy as np
import datetime
import skimage.segmentation as seg
import os
from applications.atari.custom_lime import rectangle_segmentation


def test_one_approach(saliency_fn, model, save_dir, _GAME):
    """Do the test for one saliency method."""
    # create save dir if it does not exist yet
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # if false, use a different start to introduce randomnes
    fixed_start = True

    # for reward calculation
    total_reward = 0
    reward_list = []

    np.random.seed(42)

    steps = 1001

    model.summary()
    if _GAME == "pacman":
        env = gym.make('MsPacmanNoFrameskip-v4')
    elif _GAME == "breakout":
        env = gym.make("BreakoutNoFrameskip-v4")
    elif _GAME == "spaceInvaders":
        env = gym.make("SpaceInvadersNoFrameskip-v4")
    elif _GAME == "frostbite":
        env = gym.make("FrostbiteNoFrameskip-v4")
    env.reset()
    wrapper = atari_wrapper(env)
    if (_GAME == "spaceInvaders") | (_GAME == "breakout"):
        wrapper.fire_reset = True
    wrapper.reset(noop_max=1)

    scores = []
    times = []

    if fixed_start:
        if _GAME == "pacman":
            wrapper.fixed_reset(200, 0)  # the MsPacman game does react to the first actions
        else:
            wrapper.fixed_reset(1, 0)
    for _ in range(steps):
        if _ < 4:
            action = env.action_space.sample()
            # to have more controll over the fixed starts
            if fixed_start:
                if _GAME == "breakout":
                    action = 1  # this makes breakout start much faster
                else:
                    action = 0
        else:
            my_input = np.expand_dims(stacked_frames, axis=0)
            output = model.predict(
                my_input)  # this output corresponds with the output in baseline if --dueling=False is correctly set for baselines.
            # save model predictions

            action = np.argmax(np.squeeze(output))

            # # here you can get the screen frames
            # for i in range(len(observations)):
            #     index = str(_) + '_' + str(i)
            #     observation = observations[i]

            if _ % 10 == 0:
                print("Step: " + str(_))

            tmp_scores, time = my_explainer.insertion_metric(_, saliency_fn=saliency_fn, stacked_frames=stacked_frames)
            scores.append(tmp_scores)
            times.append(time)


        stacked_frames, observations, reward, done, info = wrapper.step(action)
        total_reward += reward
        if done:
            print('total_reward', total_reward)
            reward_list.append(total_reward)
            total_reward = 0
        # env.render()

        if _ % 100 == 0 and _ != 0:
            print("Saving progress...")
            np.save(file=os.path.join(save_dir, "pred_" + str(_)), arr=scores)
            np.save(file=os.path.join(save_dir, "times_" + str(_)), arr=times)
            scores = []
            times = []

    reward_list.append(total_reward)
    average_reward = np.mean(reward_list)
    print(average_reward)
    env.close()


if __name__ == '__main__':

    for GAME in ["pacman", "breakout", "spaceInvaders", "frostbite"]:

        # model needs to be defined outside of the insertion metric since it is needed for the different saliency fn
        if GAME == "pacman":
            model_ = keras.models.load_model('../models/MsPacman_5M_ingame_reward.h5')
        elif GAME == "breakout":
            model_ = keras.models.load_model('../models/BreakoutIngame_5M.h5')
        elif GAME == "spaceInvaders":
            model_ = keras.models.load_model('../models/SpaceInvadersIngame_5M.h5')
        elif GAME == "frostbite":
            model_ = keras.models.load_model('../models/FrostbiteIngame_5M.h5')

        my_explainer = explainer(model=model_)

        dir = os.path.join("results", GAME)

        save_dir_ = os.path.join(dir, "rise_08_18_3000")
        saliency_fn_ = (lambda x: (my_explainer.generate_rise_prediction(x,
                                                                         probability=0.8,
                                                                         use_softmax=True,
                                                                         number_of_mask=3000,
                                                                         mask_size=18)))
        test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME)


        save_dir_ = os.path.join(dir, "Lime_slic_80_10_05_1000")
        if not os.path.isdir(save_dir_):
            os.makedirs(save_dir_)
        segmentation_fn = (lambda x: seg.slic(x, n_segments=80, compactness=10, sigma=0.5))
        saliency_fn_ = (lambda x: (my_explainer.generate_lime_explanation(input=x,
                                                                          hide_img=False,
                                                                          positive_only=True,
                                                                          segmentation_fn=segmentation_fn,
                                                                          num_samples=1000))[2])
        test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME)

        save_dir_ = os.path.join(dir, "Lime_quickshift_1_4_0_3000")
        segmentation_fn = (lambda x: seg.quickshift(x, kernel_size=1, max_dist=4, ratio=0, convert2lab=False))
        saliency_fn_ = (lambda x: (my_explainer.generate_lime_explanation(input=x,
                                                                          hide_img=False,
                                                                          positive_only=True,
                                                                          segmentation_fn=segmentation_fn,
                                                                          num_samples=3000))[2])
        test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME)

        save_dir_ = os.path.join(dir, "Lime_felzenswalb_1_025_2_2500")
        segmentation_fn = (
            lambda x: seg.felzenszwalb(x, scale=1, sigma=0.25, min_size=2))
        saliency_fn_ = (lambda x: (my_explainer.generate_lime_explanation(input=x,
                                                                          hide_img=False,
                                                                          positive_only=True,
                                                                          segmentation_fn=segmentation_fn,
                                                                          num_samples=2500))[2])
        test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME)


        save_dir_ = os.path.join(dir, "occl_4_0")
        saliency_fn_ = (lambda x: (my_explainer.generate_occlusion_explanation(input=x, patch_size=4, color=0,
                                                                               use_softmax=True)))
        test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME)

        save_dir_ = os.path.join(dir, "occl_4_gray")
        saliency_fn_ = (lambda x: (my_explainer.generate_occlusion_explanation(input=x, patch_size=4, color=0.5,
                                                                               use_softmax=True)))
        test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME)


        save_dir_ = os.path.join(dir, "noise_4_blur")
        saliency_fn_ = (lambda x: my_explainer.generate_greydanus_explanation(x, r=4, blur=True))
        test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME)

        save_dir_ = os.path.join(dir, "noise_4_black")
        saliency_fn_ = (lambda x: my_explainer.generate_greydanus_explanation(x, r=4, blur=False))
        test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME)

        save_dir_ = os.path.join(dir, "noise_4_blur_rawDiff")
        saliency_fn_ = (lambda x: my_explainer.generate_greydanus_explanation(x, r=4, blur=True, raw_diff=True))
        test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME)




