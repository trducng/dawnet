# Utility functions to work with images
# @author: _john
# =============================================================================
import os
import math

import cv2
import matplotlib

if os.name == 'posix' and 'DISPLAY' not in os.environ:
    # 'agg' backend for headless server (not connected to any display)
    matplotlib.use('agg')

import matplotlib.pyplot as plt


def show_images(image_list, label_list=None, max_columns=10, notebook=False):
    """Show list of images

    # Arguments
        image_list [list of np array]: list of images
        label_list [list of strings]: list of labels
        max_columns [int]: the maximum number of images to view side-by-side
        notebook [bool]: whether this function is called inside a notebook
    """
    if label_list is not None:
        if not isinstance(label_list, list):
            raise ValueError('`label_list` should be list')
        if len(image_list) != len(label_list):
            raise ValueError(
                '`image_list` should have the same length with `label_list`')

    if not isinstance(image_list, list):
        image_list = [image_list]

    columns = min(max_columns, len(image_list))
    rows = math.ceil(len(image_list) / columns)

    plt.figure(figsize=(20,10))
    for _idx, each_img in enumerate(image_list):
        plt.subplot(rows, columns, _idx+1)
        if label_list is not None:
            plt.title(label_list[_idx])
        plt.imshow(each_img, cmap='gray')

    if not notebook:
        plt.show()


def show_image_ascii(image, bgr=False):
    """Show the image in ascii

    # Arguments
        image [str or np array]: if type string, then it should be a path
            to the image
        bgr [bool]: whether the image has color channels BGR
    """
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        if len(image.shape) == 3:
            mode = cv2.COLOR_RGB2GRAY
            if bgr:
                mode = cv2.COLOR_BGR2GRAY
            image = cv2.cvtColor(image, mode)

    # Calculate image width and height
    height, width = image.shape
    columns, _ = os.get_terminal_size()
    rows = int(columns * height / width)

    # Transform and binarize image
    image = cv2.resize(image, (columns, rows), interpolation=cv2.INTER_CUBIC)
    image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Draw image into string
    new_image = ['_' * (columns)]
    for each_row in image:
        col = []
        for each_col in each_row:
            if each_col == 0:
                col.append('#')
            else:
                col.append(' ')
        new_image.append(''.join(col))
    new_image.append('_' * (columns))

    # Print out
    for each_row in new_image:
        print(each_row)
