# All functions that support visualization
# @author: _john
# ============================================================================
import os
from collections import defaultdict

import matplotlib

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
