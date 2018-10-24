# Calculate common statistics
# @author: _john
# ==============================================================================
import os
from collections import defaultdict

import matplotlib

if os.name == 'posix' and 'DISPLAY' not in os.environ:
    # 'agg' backend for headless server (not connected to any display)
    matplotlib.use('agg')

import matplotlib.pyplot as plt
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


def draw_history(history, attributes=None):
    """Draw the training progress based on history
    
    # Arguments
        history [list of objs]: list of historical measurements
        attributes [str or list of strs]: the attribute to draw on
    """
    iteration_attr = 'itr'
    if len(history) == 0:
        return

    if isinstance(attributes, str):
        attributes = [attributes]
    elif attributes is None:
        attributes = []
        for each_key in history[0].keys():
            if each_key == iteration_attr:
                continue
            attributes.append(each_key)
    elif not isinstance(attributes, list):
        raise AttributeError('the `attributes` argument must be list of '
                             'attribute strings')

    xs = []
    ys = defaultdict(list)

    for each_history in history:
        xs.append(each_history[iteration_attr])
        for each_attr in attributes:
            ys[each_attr].append(each_history[each_attr])
    
    for each_attr in attributes:
        plt.plot(xs, ys[each_attr])
    
    plt.legend(attributes)
    plt.show()        


def history_min(history, attribute):
    """Get the iterations that has lowest `attribute` value

    # Arguments
        history [list of objs]: list of historical measurements
        attribute [str or list of strs]: the attribute to draw on

    # Returns
        [float]: the minimum `attribute` value
        [int]: the iteration value
    """
    iteration_attr = 'itr'
    min_attr = float('inf')
    matched_iteration = None

    for each_history in history:
        if each_history[attribute] < min_attr:
            min_attr = each_history[attribute]
            matched_iteration = each_history[iteration_attr]
    
    return min_attr, matched_iteration

