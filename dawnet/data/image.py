# Scripts to preprocess images
# @author: _john
# ==============================================================================
import os
import math

import cv2
from imgaug import augmenters as iaa
from imgaug import imgaug as ia
from imgaug.parameters import (StochasticParameter, Deterministic, Choice, 
    DiscreteUniform, Normal, Uniform)
import matplotlib

if os.name == 'posix' and 'DISPLAY' not in os.environ:
    # 'agg' backend for headless server (not connected to any display)
    matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
from scipy.ndimage import rotate
from skimage.morphology import skeletonize



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


class PerspectiveTransform(iaa.PerspectiveTransform):
    """Rewrite the default perspective transform, which has random cropping"""

    def __init__(self, scale=0, cval=255, keep_size=True,
        name=None, deterministic=False, random_state=None):
        super(PerspectiveTransform, self).__init__(
            scale=scale, keep_size=keep_size, name=name, 
            deterministic=deterministic, random_state=random_state)

        self.cval = cval

    def _create_matrices(self, shapes, random_state):
        """Create the transformation matrix

        # Arguments
            shapes [list of tuples]: list of image shapes
            random_state [numpy Random state]: some random state

        # Returns
            [list of np array]: list of transformation matrices
            [list of ints]: list of heights
            [list of ints]: list of widths
        """
        matrices = []
        max_heights = []
        max_widths = []
        nb_images = len(shapes)
        seeds = ia.copy_random_state(random_state).randint(
            0, 10**6, (nb_images,))

        for _idx in range(nb_images):
            height, width = shapes[_idx][:2]

            pts1 = np.float32([
                [0, 0], [0, height-1], [width-1, 0], [width-1, height-1]
            ])

            transition = self.jitter.draw_samples((4, 2),
                random_state=ia.new_random_state(seeds[_idx]))
            transition[:,0] = transition[:,0] * np.min([height, width])
            transition[:,1] = transition[:,1] * np.min([height, width])
            transition = transition.astype(np.int32)
            transition[:,0] = transition[:,0] + np.abs(np.min(transition[:,0]))
            transition[:,1] = transition[:,1] + np.abs(np.min(transition[:,1]))

            pts2 = np.array([
                [transition[0,0], transition[0,1]],
                [transition[1,0], height-1+transition[1,1]],
                [width-1+transition[2,0], transition[2,1]],
                [width-1+transition[3,0], height-1+transition[3,1]]
            ], dtype=np.float32)

            height = np.max(pts2[:,1])
            width = np.max(pts2[:,0])

            matrices.append(cv2.getPerspectiveTransform(pts1, pts2))
            max_heights.append(height)
            max_widths.append(width)

        return matrices, max_heights, max_widths

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        if not self.keep_size:
            result = list(result)

        matrices, max_heights, max_widths = self._create_matrices(
            [image.shape for image in images],
            random_state
        )

        for i, (M, max_height, max_width) in enumerate(zip(matrices, max_heights, max_widths)):
            # cv2.warpPerspective only supports <=4 channels
            #ia.do_assert(images[i].shape[2] <= 4, "PerspectiveTransform is currently limited to images with 4 or less channels.")
            nb_channels = images[i].shape[2]
            if nb_channels <= 4:
                warped = cv2.warpPerspective(
                    images[i], M, (max_width, max_height), borderValue=self.cval)
                if warped.ndim == 2 and images[i].ndim == 3:
                    warped = np.expand_dims(warped, 2)
            else:
                # warp each channel on its own, re-add channel axis, then stack
                # the result from a list of [H, W, 1] to (H, W, C).
                warped = [
                    cv2.warpPerspective(
                        images[i][..., c], M, (max_width, max_height),
                        borderValue=self.cval)
                    for c in range(nb_channels)]
                warped = [warped_i[..., np.newaxis] for warped_i in warped]
                warped = np.dstack(warped)
            #print(np.min(warped), np.max(warped), warped.dtype)
            if self.keep_size:
                h, w = images[i].shape[0:2]
                warped = ia.imresize_single_image(warped, (h, w), interpolation="cubic")
            result[i] = warped

        return result


class ItalicizeLine(iaa.meta.Augmenter):
    """
    Drop-in replace for shear transformation in iaa.Affine (the implementation
    inside iaa.Affine crop images while italicize)
    """
    def __init__(self, shear=(-40, 41), cval=255, vertical=False, 
        name=None, deterministic=False, random_state=None):
        """Initialize the augmentator

        # Arguments
            shear [float or tuple of 2 floats]: if it is a single number, then
                image will be sheared in that degree. If it is a tuple of 2
                numbers, then the shear value will be chosen randomly
            cval [int]: fill-in value to new pixels
        """
        super(ItalicizeLine, self).__init__(name=name,
            deterministic=deterministic, random_state=random_state)

        if isinstance(shear, StochasticParameter):
            self.shear = shear
        elif ia.is_single_number(shear):
            self.shear = Deterministic(shear)
        elif ia.is_iterable(shear):
            ia.do_assert(
                len(shear) == 2, 
                "Expected rotate tuple/list with 2 entries, got {} entries."
                    .format((len(shear))))
            ia.do_assert(
                all([ia.is_single_number(val) for val in shear]),
                "Expected floats/ints in shear tuple/list.")
            self.shear = Uniform(shear[0], shear[1])
        else:
            raise Exception(
                "Expected float, int, tuple/list with 2 entries or "
                "StochasticParameter. Got {}.".format(type(shear)))

        self.cval = cval
        self.vertical = vertical

    def _augment_images(self, images, random_state, parents, hooks):
        """Augment the images

        # Arguments
            images [list of np array]: the list of images

        # Returns
            [list of np array]: the list of augmented images
        """
        result = images

        seed = random_state.randint(0, 10**6, 1)[0]
        shear_values = self.shear.draw_samples((len(images),),
            random_state=ia.new_random_state(seed + 80))

        for _idx, image in enumerate(result):
            angle = shear_values[_idx]
            if angle == 0:
                continue

            if self.vertical:
                # use horizontal italicization method
                image = rotate(image, -90, order=1, cval=self.cval)

            height, original_width, _ = image.shape
            distance = int(height * math.tan(math.radians(math.fabs(angle))))

            if angle > 0:
                point1 = np.array(
                    [[0, 0], [0, height], [5, 0]], dtype=np.float32)
                point2 = np.array(
                    [[distance, 0], [0, height], [5 + distance, 0]],
                    dtype=np.float32)
                image = np.concatenate(
                    [image,
                     np.ones((height,distance,1),dtype=np.uint8) * self.cval],
                    axis=1)
            else:
                point1 = np.array(
                    [[distance, 0], [distance, height], [distance + 5, 0]], 
                    dtype=np.float32)
                point2 = np.array([[0, 0], [distance, height], [5, 0]],
                    dtype=np.float32)
                image = np.concatenate(
                    [np.ones((height,distance,1),dtype=np.uint8) * self.cval,
                     image],
                    axis=1)

            height, width, _ = image.shape
            matrix = cv2.getAffineTransform(point1, point2)
            image = cv2.warpAffine(image, matrix, (width, height),
                borderValue=self.cval)
            
            if self.vertical:
                # use horizontal intalicization method
                image = rotate(image, 90, order=1, cval=self.cval)

            if image.ndim == 2:
                image = image[..., np.newaxis]

            result[_idx] = image

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
        hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.shear, self.cval]


class RotateLine(iaa.meta.Augmenter):
    """
    Drop-in replace for imgaug's Affine's rotation as the supplied rotation
    does not support fill in cval. With probability 60%
    """
    def __init__(self, angle=(-10, 10), cval=255, name=None,
        deterministic=False, random_state=None):
        """Initialize the augmentator

        # Arguments
            angle [float or tuple of 2 floats]: if it is a single number, then
                image will be rotated in that degree. If it is a tuple of 2
                numbers, then the angle value will be chosen randomly
            cval [int]: fill-in value to new pixels
        """
        super(RotateLine, self).__init__(name=name,
            deterministic=deterministic, random_state=random_state)

        if isinstance(angle, StochasticParameter):
            self.angle = angle
        elif ia.is_single_number(angle):
            self.angle = Deterministic(angle)
        elif ia.is_iterable(angle):
            ia.do_assert(
                len(angle) == 2, 
                "Expected rotate tuple/list with 2 entries, got {} entries."
                    .format((len(angle))))
            ia.do_assert(
                all([ia.is_single_number(val) for val in angle]),
                "Expected floats/ints in angle tuple/list.")
            self.angle = Uniform(angle[0], angle[1])
        else:
            raise Exception(
                "Expected float, int, tuple/list with 2 entries or "
                "StochasticParameter. Got {}.".format(type(angle)))

        self.cval = cval

    def _augment_images(self, images, random_state, parents, hooks):
        """Augment the images

        # Arguments
            images [list of np array]: the list of images

        # Returns
            [list of np array]: the list of augmented images
        """
        result = images

        seed = random_state.randint(0, 10**6, 1)[0]
        angle_values = self.angle.draw_samples((len(images),),
            random_state=ia.new_random_state(seed + 90))

        for _idx, image in enumerate(result):
            angle = angle_values[_idx]
            if angle == 0:
                continue
            result[_idx] = rotate(image, angle, order=1, cval=self.cval)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
        hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.angle, self.cval]


class PencilStroke(iaa.meta.Augmenter):
    """
    Transform the image to have pencil stroke
    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        """Initialize the augmentator"""
        super(PencilStroke, self).__init__()

    def _augment_images(self, images, random_state, parents, hooks):
        """Construct pencil stroke

        This method works by binarize an image, and then transform the stroke
        to have a Gaussian distribution around some mean and with certain
        distribution.

        # Arguments
            image [np array]: the character image that we will transform
                stroke to pencil stroke. This image is expected to have dark
                strokes on white background

        # Returns
            [np array]: the line text image with pencil stroke
        """
        result = images

        for _idx, each_image in enumerate(images):
            image = each_image[:,:,0]

            # mean pixel in range 140 - 170, distribution in range 0.08 - 0.2
            mean_pixel = random.choice(30) + 140
            distribution = (random.choice(12) + 8) / 100

            # Binarize and invert the image to have foreground 1 & background 0
            bin_image = cv2.threshold(
                image, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            reconstruct_image = (
                255 - (bin_image * (255 - mean_pixel))).astype(np.float32)

            # Create a Gaussian kernel
            kernel_scale = random.choice(2) + 2
            small_shape = (int(image.shape[0]/kernel_scale),
                           int(image.shape[1]/kernel_scale))
            large_shape = (image.shape[1], image.shape[0])  # width x height
            kernel = cv2.resize(
                random.normal(0, distribution, small_shape),
                large_shape,
                interpolation=cv2.INTER_NEAREST).astype(np.float32)

            image = reconstruct_image + reconstruct_image * bin_image * kernel
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = cv2.blur(image, (2,2))
            if random.random() < 0.3:
                image = cv2.blur(image, (2,2))

            image = image[..., np.newaxis]
            result[_idx] = image

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
        hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return []


class Skeletonize(iaa.meta.Augmenter):
    """
    Randomly skeletonize the image
    """
    def __init__(self, is_binary, name=None, deterministic=False,
        random_state=None):
        """Initialize the augmentator"""
        super(Skeletonize, self).__init__()

        self.is_binary = is_binary

    def _augment_images(self, images, random_state, parents, hooks):
        """Augment the images

        # Arguments
            images [list of np array]: the list of images

        # Returns
            [list of np array]: the list of augmented images
        """
        result = images

        for _idx, each_image in enumerate(images):
            image = each_image[:,:,0]

            dilation_value = 5 if random.random() <= 0.5 else 3
            image = skeletonize_image(image, dilation_value, self.is_binary)

            image = image[..., np.newaxis]
            result[_idx] = image

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
        hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.is_binary]


def augment_image(image, color_bg=255, italicize=0, angle=0, pad_vertical=0,
        pad_horizontal=0, pt=0, elas_alpha=0, elas_sigma=0.4, blur_type=0,
        blur_value=0, brightness=1.0, gauss_noise=0):
    """Augment the image using provided attributes"""
    augmentators = []

    # must have background color
    # italicize range (-30, 30, float, by 0.5)
    if italicize != 0:
        augmentators.append(ItalicizeLine(shear=italicize, cval=color_bg))
    
    # rotate range (-10, 10, float, by 0.5)
    if angle != 0:
        augmentators.append(RotateLine(angle=angle, cval=color_bg))

    # add padding here (maybe just the percent option)
    if pad_vertical != 0:
        augmentators.append(iaa.Pad(percent=(pad_vertical, 0, pad_vertical, 0),
                                    pad_mode='constant', pad_cval=color_bg))
    if pad_horizontal != 0:
        augmentators.append(iaa.Pad(
            percent=(0, pad_horizontal, 0, pad_horizontal),
            pad_mode='constant', pad_cval=color_bg))

    # perspective transform range (0, 0.3, float, 0.02)
    if pt != 0:
        augmentators.append(PerspectiveTransform(
            scale=pt, cval=color_bg, keep_size=False))
    
    # elastic alpha (0, 1.0, float, 0.05), sigma (0.4, 0.6, float, 0.05)
    if elas_alpha != 0:
        elas_sigma = max(elas_sigma, 0.4)
        augmentators.append(iaa.ElasticTransformation(
            alpha=elas_alpha, sigma=elas_sigma, cval=color_bg))
    
    # blur_type: 0, 1, 2, 3 -> None, Gaussian, Average, Median, blur_value
    # either (0.0, 1.5, float, 0.1) for Gaussian or (1, 5, int, 2) for other
    if blur_type == 1:
        blur_value = blur_value / 10
        augmentators.append(iaa.GaussianBlur(blur_value))
    elif blur_type == 2:
        blur_value = max(blur_value, 1)
        augmentators.append(iaa.AverageBlur(blur_value))
    elif blur_type == 3:
        blur_value = max(blur_value, 1)
        augmentators.append(iaa.MedianBlur(blur_value))
    
    # brightness (0.3, 1.8, float, 0.05)
    if brightness != 1:
        augmentators.append(iaa.Multiply(brightness))
    
    # gaussian noise (0.0, 0.2, float, 0.01)
    if gauss_noise != 0:
        augmentators.append(iaa.AdditiveGaussianNoise(
            scale=gauss_noise * color_bg))

    return iaa.Sequential(augmentators).augment_image(image)
