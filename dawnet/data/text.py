# Text data operations
# @author: _john
# =============================================================================
import difflib
from functools import partial

import colorama
from termcolor import colored

from dawnet.utils.dependencies import print_md, colored_md


def view_string_prediction(prediction, ground_truth, to_print=True,
    notebook=True):
    """Prettify string prediction

    Incorrect characters in the prediction will be marked red, while correct
    one will be marked green

    # Arguments
        prediction [str]: the predicted string
        ground_truth [str]: the ground truth string
        to_print [bool]: whether to pretty print it out in the terminal

    # Returns
        [list of tuples of 2]: the first element of each tuple is a character
            string and the second element is boolean denoting whether that
            character is correct
    """

    result_pred = []
    for each_result in difflib.ndiff(ground_truth, prediction):
        if each_result[0] == ' ':
            result_pred.append((each_result[-1], True))
        elif each_result[0] == '+':
            result_pred.append((each_result[-1], False))

    result_ground = []
    for each_result in difflib.ndiff(prediction, ground_truth): # merely switch
        if each_result[0] == ' ':
            result_ground.append((each_result[-1], True))
        elif each_result[0] == '+':
            result_ground.append((each_result[-1], False))

    if to_print:
        color_func = colored_md if notebook else partial(colored,attrs=['bold'])
        content = ''
        for each_result in result_ground:
            if each_result[1]:
                content += color_func(each_result[0])
            else:
                content += color_func(each_result[0], 'green')

        content += color_func(' ==> ')
        
        for each_result in result_pred:
            if each_result[1]:
                content += color_func(each_result[0])
            else:
                content += color_func(each_result[0], 'red')

        if notebook:
            print_md('<b>{}</b>'.format(content))
        else:
            colorama.init()
            print(content)

    return result_pred, result_ground
