''' Utility functions used in the evaluations'''

import numpy as np


def process_single_insertion_result(arr):
    og_prediction = arr[-1]
    og_action = np.argmax(og_prediction)
    insertion_result = np.empty(len(arr))
    for i in range(len(arr)):
        insertion_result[i] = advantage(arr[i], og_action)
        # insertion_result[i] = arr[i][og_action]
    # normalize such that the result goes from 0 for the fully perturbed image to 1 for the full image
    # insertion_result = normalize(insertion_result)
    return insertion_result


def advantage(q_values, index):
    ''' Calculates the advantage of the action with the given *index* compared to the other *q-vals*'''
    observed_action_value = q_values[index]
    action_value_mean = np.mean(q_values)
    advantage = observed_action_value - action_value_mean
    return advantage


# Normalizing formula from https://arxiv.org/pdf/2001.00396.pdf
def normalize(arr):
    out = []
    b = arr[0]
    t_1 = arr[-1]
    if b > t_1:
        print("THIS SHOULD NOT HAPPEN! The perturbation image value is higher then the one of the original image.")
    for val in arr:
        out.append((val - b)/(t_1 - b))
    out = np.asarray(out)
    return out


def auc(arr):
    """
    simple Area Under the curve calculation using sum_of_values/number_of_values
    :param arr:
    :return:
    """
    # arr2 = arr - arr[0]
    auc = arr.sum() / (arr.shape[0])
    print(round(auc,3))
    return auc


def aoc(arr):
    """
    simple Area Over the curve.
    :param arr:
    :return:
    """
    arr2 = -1 * arr + arr[-1]
    return arr2.sum() / (arr2.shape[0])

