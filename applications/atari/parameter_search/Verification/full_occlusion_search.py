"""
Module for doing a full parameter search for 1000 states of the game Pacman with fast parameter configurations of occlusion.
This can then be used to verify which states are suited to search for good parameters.

This script needs to be run from the insertion_metric folder, since the insertion results are saved there.
These results are then processed by full_occlusion_evaluation.py.
"""

from applications.atari.explanation import explainer
import applications.atari.rise as rise
import keras
import numpy as np
import os

from applications.atari.insertion_metric.insertion_metric_main import test_one_approach


def random_baseline(stacked_input):
    return np.random.uniform(low=0.0, high=1.0, size=[stacked_input.shape[0], stacked_input.shape[1]])


if __name__ == '__main__':
    for insertion_color in ["random", "black"]:

        if insertion_color == "random":
            insertion_fn = rise.random_occlusion
        elif insertion_color == "black":
            insertion_fn = rise.custom_black

        for GAME in ["pacman"]:

            for i in range(4, 11):
                for j in range(0, 2):
                    for k in range(0, 2):
                        patch_size = i
                        color = 0.5 * j
                        softmax = k
                        parameters = (patch_size, color, softmax)

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

                        name = "occl_" + str(patch_size) + '_' + str(color) + '_' + str(softmax)
                        save_dir_ = os.path.join(dir, name)
                        saliency_fn_ = (lambda x: (my_explainer.generate_occlusion_explanation(input=x, patch_size=patch_size, color=color,
                                                                                               use_softmax=softmax)))
                        test_one_approach(saliency_fn=saliency_fn_, model=model_, save_dir=save_dir_, _GAME=GAME,
                                          insertion_substrate_fn= insertion_fn)



