# Scripts to preprocess images
# @author: _john
# ==============================================================================
import cv2
import numpy as np
import torchvision


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
