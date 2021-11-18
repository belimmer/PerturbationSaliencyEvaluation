"""This module is only used to generate random sets of 10 pacman states in order to test them as potential datasets for
the parameter search."""

import os
from shutil import copy
import random


def choose_random_states(output_dir, stream_dir,number_of_states):
    # choose random state indices
    summary_states = random.sample(range(1, 10000), number_of_states)

    # copy the chosen states into another directory
    if not (os.path.exists(output_dir)):
        os.mkdir(output_dir)
    for state_index in summary_states:
        # copy the screens
        # each state is comprised of 4 skipped frames
        for i in range(4):
            screen_name = "screen_" + str(state_index) + "_" + str(i) + ".png"
            source_file = os.path.join(stream_dir, "screen", screen_name)
            dst_file = os.path.join(output_dir, screen_name)
            copy(source_file, dst_file)
        # copy the state
        state_name = "state_" + str(state_index) + ".npy"
        source_file = os.path.join(stream_dir, "state", state_name)
        dst_file = os.path.join(output_dir, state_name)
        copy(source_file, dst_file)
        state_name = "state_" + str(state_index) + ".png"
        source_file = os.path.join(stream_dir, "state", state_name)
        dst_file = os.path.join(output_dir, state_name)
        copy(source_file, dst_file)


if __name__ == '__main__':
    base_out_dir = "random_states"
    stream_dir_ = "../../stream"

    for i in range(10):
        out_dir = base_out_dir + "_" + str(i)
        choose_random_states(out_dir, stream_dir_, 10)
