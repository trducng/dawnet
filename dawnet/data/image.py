# Scripts to preprocess images
# @author: _john
# ==============================================================================
import os
import math
from urllib.request import urlopen
from typing import List, Union

import cv2
import matplotlib
import numpy as np
from IPython.display import display
from ipywidgets import widgets, Box
from PIL import Image

if os.name == 'posix' and 'DISPLAY' not in os.environ:
    # 'agg' backend for headless server (not connected to any display)
    matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import numpy.random as random
from scipy.ndimage import rotate
from skimage.morphology import skeletonize


def download_image(url):
    """Download an image to numpy array

    # Arguments
        url [str]: the url

    # Returns
        [np array]: the image
    """
    response = urlopen(url)
    image = np.asarray(bytearray(response.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def get_rectangle_vertices(pixels):
    """Given a list of pixels of a rectangle, retrieve the 4 rectangle corners

    # Arguments
        pixels [list of tuples of 2 ints]: the list of pixels in y, x

    # Returns
        [list of 4 ints]: coordinate of top, bottom, left, right
    """
    pixels = sorted(pixels, key=lambda obj: obj[0])
    pixels = sorted(pixels, key=lambda obj: obj[1])

    top, left = pixels[0]
    bottom, right = pixels[-1]

    return top, bottom, left, right


## Jupyter notebooks
def show_images(images, label_list=None, max_columns=10, show=True, output=None):
    """Show grid of images

    Args:
        image_list <[np array]>: list of images
        label_list <[str]>: list of labels
        max_columns <int>: the maximum number of images to view side-by-side
        show <bool>: whether this function is called inside a notebook
        output <str>: if set, save the result to this path
    """
    if label_list is not None:
        if not isinstance(label_list, list):
            raise ValueError('`label_list` should be list')
        if len(images) != len(label_list):
            raise ValueError(
                '`images` should have the same length with `label_list`')

    if isinstance(images, np.ndarray):
        images = [images[idx] for idx in range(images.shape[0])]

    columns = min(max_columns, len(images))
    rows = math.ceil(len(images) / columns)

    plt.figure(figsize=(columns*5,rows*5), facecolor='white')
    for _idx, each_img in enumerate(images):
        plt.subplot(rows, columns, _idx+1)
        if label_list is not None:
            plt.title(label_list[_idx])
        plt.imshow(each_img, cmap='gray')

    plt.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])    # remove ticks
    if show:
        plt.show()
    if output:
        plt.savefig(output)
        plt.close()


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


def show_animation(images, *args, **kwargs):
    """Quickly show animation from list of images

    Inside jupyter notebook, should have `%matplotlib notebook`
    Several good approaches can be found:
    https://stackoverflow.com/questions/35532498/animation-in-ipython-notebook

    # Arguments
        images [list of 2D or 3D np array]: each image is H x W x C
        *args, **kwargs: other params for function. Some good arguments are:
            interval [int]: delay in milliseconds, default 200
            repeat_delay [int]: delay between repeat, in milliseconds
            repeat [bool]: whether to repeat, default True
    """
    if not images:
        return

    fig = plt.figure(figsize=(20, 10))
    im = plt.imshow(images[0], animated=True)
    animate = lambda idx: im.set_array(images[idx])

    ani = animation.FuncAnimation(fig, animate, frames=len(images), *args, **kwargs)
    plt.show()
    plt.close()


def paste_image(original_image, pasted_image, original_portion, pasted_portion):
    """Cut a rectangular portion of `original_image` to `pasted_image`

    # Arguments
        original_image [nd array]: the original image to cut from
        pasted_image [nd array]: the pasted image
        original_portion [tuple of 4 ints]: top, bottom, left, right
        pasted_portion [tuple of 2 ints]: top, left

    # Returns
        [nd array]: the new `pasted_iamge` with portion replaced by
            `original_image`
    """
    new_image = np.copy(pasted_image)

    # get the positions
    o_top, o_bottom, o_left, o_right = original_portion
    p_top, p_left = pasted_portion
    p_bottom, p_right = p_top + (o_bottom - o_top), p_left + (o_right - o_left)

    new_image[p_top:p_bottom,
              p_left:p_right] = original_image[o_top:o_bottom,o_left:o_right]

    return new_image


def get_subplot_rows_cols(tensor, fixed_rows=None, fixed_cols=None):
    """The the suitable number of rows and columns for subplotting

    The suitable number of images must the one that makes the overal figure
    square.

    condition  columns   rows
    == 1       -> 5        5
    <= 2       -> 4        4
    <= 4       -> 3        4
    <= 6       -> 2        4
                  1        4

    # Argument
        tensor [3D np or torch tensor]: the tensor to view subplots

    # Returns
        [tuple of 2 ints]: the number of rows and columns
    """
    height, width = tensor.shape[-2:]

    reverse = False
    if height > width:
        width, height = height, width
        reverse = True

    if width // height == 1:
        columns, rows = 5, 5
    elif width // height <= 2:
        columns, rows = 4, 5
    elif width // height <= 4:
        columns, rows = 3, 4
    elif width // height <= 6:
        columns, rows = 2, 4
    else:
        columns, rows = 1, 4

    if reverse:
        columns, rows = rows, columns

    rows = rows if fixed_rows is None else fixed_rows
    columns = columns if fixed_cols is None else fixed_cols

    return rows, columns


## Image manipulation
def resize(image, width=None, height=None):
    """Resize the image to match width and height

    If any of the width or heigth is missing, then the image is rescaled
    to have the size of the given height or width.

    # Arguments
        image [np array]: the image
        width [int]: the width
        height [int]: the height

    # Returns
        [np array]: the resized image
    """
    if width is None and height is None:
        raise AttributeError('either `width` or `height` must be given')

    if width is not None and height is not None:
        return cv2.resize(image, (width, height), cv2.INTER_LINEAR)

    height_original, width_original = image.shape[:2]
    if width is None:
        width = int(height * width_original / height_original)

    if height is None:
        height = int(width * height_original / width_original)

    return cv2.resize(image, (width, height), cv2.INTER_LINEAR)


def smoothen_image(image, sampling_level=4, is_binary=False):
    """Smoothen the image (to reduce zagged elements)

    # Arguments
        image [np array]: a binary image (1: background, 0: foreground)
        sampling_level [int]: the higher the sampling level, the smoother the
            dilation can be, default to 4, maximum 5 (higher value will cost
            more memory)

    # Returns
        [np array]: an image with width normalized to `pixel` level
    """
    max_pixel = 1 if is_binary else 255
    sampling_level = int(min(sampling_level, 5))
    sampling_level = int(max(sampling_level, 1))

    image = max_pixel - image

    # smoothen the edges
    for _ in range(sampling_level):
        image = cv2.pyrUp(image)

    for _ in range(4):
        image = cv2.medianBlur(image, 3)

    for _ in range(sampling_level):
        image = cv2.pyrDown(image)

    # revert the image
    image = max_pixel - image

    return image.astype(np.uint8)


def skeletonize_image(image, pixel, is_binary):
    """Skeletonize the image

    # Arguments
        image [np array]: a binary image (1: background, 0: foreground)
        pixel [int]: a pre-defined character width
        is_binary [bool]: whether the input image is a binary image

    # Returns
        [np array]: an image with width normalized to `pixel` level
    """
    if not is_binary:
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        image = (image / 255).astype(np.uint8)

    # skeletonize the image
    image = 1 - image
    image = skeletonize(image)
    image = image.astype(np.uint8)

    # dilate the image to have `pixel` width
    if pixel > 1:
        kernel = np.ones((pixel, pixel), dtype=np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
    image = 1 - image

    # convert image to 8-bit to smoothen and binarize if possible
    image = image * 255
    image = smoothen_image(image, is_binary=False)

    if is_binary:
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        image = (image / 255).astype(np.uint8)

    return image.astype(np.uint8)


def image_carousel(image_list):
    """Construct image carousel for Jupyter Notebook

    Example:
        ```
        from Ipython.display import display

        buttons, output = image_carousel(image_list)
        display(buttons)
        display(output)
        ```

    # Args
        image_list <[np array]>: list of images

    # Returns
        <Box>: ipywidgets box containing prev and next buttons
        <Output>: ipywidgets output
    """
    prev_button = widgets.Button(description="Prev")
    next_button = widgets.Button(description="Next")
    current = {'current': 0}

    output = widgets.Output()
    with output:
        img = image_list[current['current']]
        display(Image.fromarray(img))


    def on_prev(b):
        with output:
            current['current'] = (current['current'] - 1) % len(image_list)
            img = image_list[current['current']]
            output.clear_output()
            display(Image.fromarray(img))

    def on_next(b):
        with output:
            current['current'] = (current['current'] + 1) % len(image_list)
            img = image_list[current['current']]
            output.clear_output()
            display(Image.fromarray(img))

    prev_button.on_click(on_prev)
    next_button.on_click(on_next)

    return Box(children=[prev_button, next_button]), output

