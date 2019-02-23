# All functions that support visualization
# @author: _john
# ============================================================================
import os
from collections import defaultdict, Counter

import matplotlib
import numpy as np

if os.name == 'posix' and 'DISPLAY' not in os.environ:
    # 'agg' backend for headless server (not connected to any display)
    matplotlib.use('agg')

import matplotlib.pyplot as plt
from dawnet.diagnose.statistics import history_max, history_min


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

    if len(attributes) == 0:
        return

    elif len(attributes) == 1:
        xs, ys = [], []
        for each_history in history:
            xs.append(each_history[iteration_attr])
            ys.append(each_history[attributes[0]])

        plt.plot(xs, ys)
        plt.title(attributes[0])
        plt.show()

    else:
        xs = list(map(lambda obj: obj[iteration_attr], history))
        ys = defaultdict(list)

        for each_attr in attributes:
            max_value, _ = history_max(history, each_attr)[0]
            min_value, _ = history_min(history, each_attr)[0]
            range_value = max_value - min_value
            for each_history in history:

                if (isinstance(each_history[each_attr], list) and
                    len(each_history[each_attr]) == 1):
                    value = each_history[each_attr][0]
                else:
                    value = each_history[each_attr]

                ys[each_attr].append((value-min_value)/range_value)

        for each_attr in attributes:
            plt.plot(xs, ys[each_attr])

        plt.legend(attributes)
        plt.show()


def histogram_counter(counter, ax=None):
    """"
    This function creates a bar plot from a counter.

    # Arguments
        counter [dict or list]: a counter of {key: count}
        ax [plt subplot]: a pyplot subplot
    """
    if isinstance(counter, list) or isinstance(counter, tuple):
        counter = Counter(counter)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    frequencies = list(counter.values())
    names = list(counter.keys())

    x_coordinates = np.arange(len(counter))
    ax.bar(x_coordinates, frequencies, align='center')

    ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(names))

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')

    return ax


def draw_1d_plot(model, param1, param2, metric):
    """Draw 1D plot from the param1 model to the param2 model

    # Arguments
        model [nn.Module]: the model
        param1 [dictionary]: the state dict
        param2 [dictionary]: the state dict
        metric [function]: a function that takes in a model and spit out a
            value

    # Returns
        [np array]: the plot
    """
    pass
