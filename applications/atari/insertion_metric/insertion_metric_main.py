"""
This module does the insertion metric experiments and measures the run-time of the saliency map generation approaches.
"""

from applications.atari.custom_atari_wrapper import atari_wrapper
from applications.atari.explanation import explainer
import applications.atari.rise as rise
import gym
import keras
import numpy as np
import skimage.segmentation as seg
import os
import timeit


def test_one_approach(saliency_fn, model, save_dir, _GAME, insertion_substrate_fn):
    """Do the insertion metric test for one saliency method."""
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

    insertion = rise.CausalMetric(model=model, mode='ins', step=84,
                                       substrate_fn= insertion_substrate_fn)  # TODO Fix dirty number for steps

    model.summary()
    if _GAME == "pacman":
        env = gym.make('MsPacmanNoFrameskip-v4')
    elif _GAME == "breakout":
        env = gym.make("BreakoutNoFrameskip-v4")
    elif _GAME == "spaceInvaders":
        env = gym.make("SpaceInvadersNoFrameskip-v4")
    elif _GAME == "frostbite":
        env = gym.make("FrostbiteNoFrameskip-v4")
    #fix the seed of the first env, such that the results stay reproducible even if more approaches are tested in one run
    env.seed(0)
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

            if _ % 10 == 0:
                print("Step: " + str(_))

            # Generate Saliency map and measure the time needed for this creation
            start = timeit.default_timer()
            saliency_map = saliency_fn(np.squeeze(stacked_frames))
            stop = timeit.default_timer()
            time = stop - start

            # Calculate Insertion metric scores for this saliency maps
            tmp_scores = insertion.single_run(img_tensor=np.squeeze(stacked_frames), explanation=saliency_map)

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


def random_baseline(stacked_input):
    return np.random.uniform(low=0.0, high=1.0, size=[stacked_input.shape[0], stacked_input.shape[1]])


if __name__ == '__main__':
    for insertion_color in ["random", "black"]:

        if insertion_color == "random":
            insertion_fn = rise.random_occlusion
        elif insertion_color == "black":
            insertion_fn = rise.custom_black

        for GAME in ["pacman", "breakout", "spaceInvaders", "frostbite"]:

            # model needs to be defined outside of the insertion metric since it is needed for the different saliency fns
            if GAME == "pacman":
                model_ = keras.models.load_model('../models/MsPacman_5M_ingame_reward.h5')
            elif GAME == "breakout":
                model_ = keras.models.load_model('../models/BreakoutIngame_5M.h5')
            elif GAME == "spaceInvaders":
                model_ = keras.models.load_model('../models/SpaceInvadersIngame_5M.h5')
            elif GAME == "frostbite":
                model_ = keras.models.load_model('../models/FrostbiteIngame_5M.h5')

            my_explainer = explainer(model=model_)

            dir = os.path.join("results", GAME, insertion_color + "_insertion")


            # RANDOM BASELINE
            save_dir_ = os.path.join(dir, "rnd_baseline")
            saliency_fn_ = random_baseline
            test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME,
                              insertion_substrate_fn= insertion_fn)

            # OCCLUSION
            PATCH_SIZE = 2
            COLOR = 0
            SOFTMAX = 0
            save_dir_ = os.path.join(dir, "occl_" + str(PATCH_SIZE) + "_" + str(COLOR) + "_" + str(SOFTMAX))
            saliency_fn_ = (lambda x: (my_explainer.generate_occlusion_explanation(input=x, patch_size=PATCH_SIZE, color=COLOR,
                                                                                   use_softmax=SOFTMAX)))
            test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME,
                              insertion_substrate_fn=insertion_fn)

            # NOISE
            RADIUS = 3
            BLUR = 1
            save_dir_ = os.path.join(dir, "noise_" + str(RADIUS) + "_" + str(BLUR))
            saliency_fn_ = (lambda x: my_explainer.generate_greydanus_explanation(x, r=RADIUS, blur=BLUR))
            test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME,
                              insertion_substrate_fn=insertion_fn)

            # SARFA
            RADIUS = 2
            BLUR = 0
            save_dir_ = os.path.join(dir, "sarfa_" + str(RADIUS) + "_" + str(BLUR))
            saliency_fn_ = (lambda x: my_explainer.generate_sarfa_explanation(x, r=RADIUS, blur=BLUR))
            test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME,
                              insertion_substrate_fn=insertion_fn)

            # RISE
            PROBABILITY = 0.5
            MASK_SIZE = 8
            NUM_MASKS = 1000
            SOFTMAX = 0
            save_dir_ = os.path.join(dir, "rise_" + str(PROBABILITY) + "_" + str(MASK_SIZE) + "_" + str(NUM_MASKS)
                                     + "_" + str(SOFTMAX))
            saliency_fn_ = (lambda x: (my_explainer.generate_rise_prediction(x,
                                                                             probability=PROBABILITY,
                                                                             use_softmax=SOFTMAX,
                                                                             number_of_mask=NUM_MASKS,
                                                                             mask_size=MASK_SIZE)))
            test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME,
                              insertion_substrate_fn=insertion_fn)

            # LIME
            # SLIC
            N_SEGMENTS = 200
            COMPACTNESS = 10
            SIGMA = 0.25
            NUM_SAMPLES = 2000
            save_dir_ = os.path.join(dir, "Lime_slic_" + str(N_SEGMENTS) + "_" + str(COMPACTNESS) + "_" + str(SIGMA)
                                     + "_" + str(NUM_SAMPLES))
            if not os.path.isdir(save_dir_):
                os.makedirs(save_dir_)
            segmentation_fn = (lambda x: seg.slic(x, n_segments=N_SEGMENTS, compactness=COMPACTNESS, sigma=SIGMA))
            saliency_fn_ = (lambda x: (my_explainer.generate_lime_explanation(input=x,
                                                                              hide_img=False,
                                                                              positive_only=True,
                                                                              segmentation_fn=segmentation_fn,
                                                                              num_samples=NUM_SAMPLES))[2])
            test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME,
                              insertion_substrate_fn=insertion_fn)

            # QUICKSHIFT
            KERNEL_SIZE = 1
            MAX_DIST = 2
            RATIO = 0.0
            NUM_SAMPLES = 1500
            save_dir_ = os.path.join(dir, "Lime_quickshift_" + str(KERNEL_SIZE) + "_" + str(MAX_DIST) + "_" +
                                     str(RATIO) + "_" + str(NUM_SAMPLES))
            segmentation_fn = (lambda x: seg.quickshift(x, kernel_size=KERNEL_SIZE, max_dist=MAX_DIST, ratio=RATIO, convert2lab=False))
            saliency_fn_ = (lambda x: (my_explainer.generate_lime_explanation(input=x,
                                                                              hide_img=False,
                                                                              positive_only=True,
                                                                              segmentation_fn=segmentation_fn,
                                                                              num_samples=NUM_SAMPLES))[2])
            test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME,
                              insertion_substrate_fn=insertion_fn)

            # FELZENSWALB (scale, sigma, min_size, num_samples)
            SCALE = 21
            SIGMA = 0.5
            MIN_SIZE = 4
            NUM_SAMPLES = 1000
            save_dir_ = os.path.join(dir, "Lime_felzenswalb_" + str(SCALE) + "_" + str(SIGMA) +
                                     "_" + str(MIN_SIZE) + "_" + str(NUM_SAMPLES))
            segmentation_fn = (
                lambda x: seg.felzenszwalb(x, scale=SCALE, sigma=SIGMA, min_size=MIN_SIZE))
            saliency_fn_ = (lambda x: (my_explainer.generate_lime_explanation(input=x,
                                                                              hide_img=False,
                                                                              positive_only=True,
                                                                              segmentation_fn=segmentation_fn,
                                                                              num_samples=NUM_SAMPLES))[2])
            test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME,
                              insertion_substrate_fn=insertion_fn)
