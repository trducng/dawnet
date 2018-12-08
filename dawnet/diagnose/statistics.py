# Calculate common statistics
# @author: _john
# ==============================================================================
import os
from collections import defaultdict

import numpy as np


def normalize_to_range(nd_array, min_value=0, max_value=255):
    """Normalize the array to hold value in certain range

    This function is handy to normalize an image into image range so that it
    can be viewed visually (that's reason default of min value and max value are
    0 and 255 respectively).

    # Arguments
        nd_array [nd array]: the original array
        min_value [int]: the minimum value (default to 0)
        max_vaue [int]: the maximum value (default to 255)

    # Returns
        [nd array]: the normalized array
    """
    current_max, current_min = np.max(nd_array), np.min(nd_array)

    nd_array = (
        (max_value - min_value)
        * (nd_array - current_min) / (current_max - current_min)
        + min_value)

    if np.max(nd_array) < 256 and np.min(nd_array) >= 0:
        # normalize into image range is a very normal case
        nd_array = nd_array.astype(np.uint8)

    return nd_array


def get_statistics(nd_array):
    """Calculate the array statistics

    Currently return the mean, median and standard deviation

    # Arguments
        nd_array [nd array]: the array to get statistics

    # Returns
        [tuple of floats]: arithmetic mean, median, std, max, min
    """
    mean = np.mean(nd_array)
    median = np.median(nd_array)
    std = np.std(nd_array)
    max_value = np.max(nd_array)
    min_value = np.min(nd_array)

    return mean, median, std, max_value, min_value


def history_min(history, attribute):
    """Get the iterations that has lowest `attribute` value

    # Arguments
        history [list of objs]: list of historical measurements
        attribute [str or list of strs]: the attribute to draw on

    # Returns
        [float]: the minimum `attribute` value
        [int]: the iteration value
    """
    xs = np.asarray(list(map(lambda obj: obj['itr'], history)),dtype=np.float32)
    ys = np.asarray(list(map(lambda obj: obj[attribute], history)),
        dtype=np.float32)
    min_values = np.argmin(ys, axis=0)

    try:
        min_values = list(min_values)
        return [(ys[each][idx].item(), xs[each])
            for idx, each in enumerate(min_values)]
    except TypeError:
        min_values = [min_values]
        return [(ys[each].item(), xs[each]) for each in min_values]


def history_max(history, attribute):
    """Get the iterations that has highest `attribute` value

    # Arguments
        history [list of objs]: list of historical measurements
        attribute [str or list of strs]: the attribute to draw on

    # Returns
        [float]: the maximum `attribute` value
        [int]: the iteration value
    """
    xs = np.asarray(list(map(lambda obj: obj['itr'], history)),dtype=np.float32)
    ys = np.asarray(list(map(lambda obj: obj[attribute], history)),
        dtype=np.float32)
    max_values = np.argmax(ys, axis=0)

    try:
        max_values = list(max_values)
        return [(ys[each][idx].item(), xs[each])
            for idx, each in enumerate(max_values)]
    except TypeError:
        max_values = [max_values]
        return [(ys[each].item(), xs[each]) for each in max_values]
