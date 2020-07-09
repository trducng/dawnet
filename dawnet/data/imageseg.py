import cv2
import numpy as np

from dawnet.utils.colormap import interpolate


def visualize_mask(mask, n_classes):
    """Visualize integer mask into RGB image

    # Args
        mask <2D np array>: the mask of type np.int32, each unique integer value
            inside the mask represents a class
        n_classes <int>: the number of classes, in cases the `mask` does not
            contain all classes

    # Returns
        <3D np array>: the visualized image in RGB space of shape W * H * 3
    """
    output = np.zeros((*mask.shape, 3)).astype(np.float32)
    for unique_value in list(np.unique(mask)):
        output[(mask == unique_value).nonzero()] = interpolate(unique_value / n_classes)

    return (output * 255).astype(np.uint8)


def create_seg_mask(labels, image):
    """Create segmentation masks

    Note:
        - the created segmask ignores iscrowd

    # Args
        labels <[{}]>: each dict contains 'category_id', 'segmentation' and 'area'.
            - 'category_id' is an integer
            - 'segmentation' is a list of lists of polygon vertices [x1 y1 x2 y2 ..]
            - 'area' is a float
        image <np array>: the image of shape [height x width x channel]

    # Returns
        <[2D np-array]>: each np array is a fill mask
        <[2D np-array]>: each np array is a border mask
        <[int]>: each int is a class
    """
    fill_masks, border_masks, categories = [], [], []
    for each_label in labels:
        segpoints = [
            np.asarray(each).reshape(-1, 2).astype(np.int32)
            for each in each_label['segmentation']
            if not each_label['iscrowd']
        ]

        full_mask = cv2.fillPoly(
                np.zeros(image.shape[:2]).astype(np.uint8), segpoints, 1
        )
        fill_mask = cv2.erode(full_mask, np.ones((3,3),np.uint8), iterations=3)
        border_mask = cv2.bitwise_xor(full_mask, fill_mask)

        fill_masks.append(fill_mask)
        border_masks.append(border_mask)
        categories.append(each_label['category_id'])

    return fill_masks, border_masks, categories


def embed_seg_mask(labels, image, n_classes=None):
    """Embed the segmentation mask on image

    The masks are generated with some criteria:
        - For overlapping masks, the smaller mask will be visualized on top

    # Args
        labels <[{}]>: each dict contains 'category_id', 'segmentation' and 'area'.
            - 'category_id' is an integer
            - 'segmentation' is a list of lists of polygon vertices [x1 y1 x2 y2 ..]
            - 'area' is a float
        image <np array>: the image of shape [height x width x channel]
        n_classes <int>: the number of classes

    # Returns
        <3D np array>: the mask-embedded image
        <2D np array>: the fill mask image
        <2D np array>: the border mask image
    """
    labels = sorted(labels, key=lambda obj: obj['area'])
    fill_masks, border_masks, categories = create_seg_mask(labels, image)
    n_classes = len(set(categories)) if n_classes is None else n_classes

    # collapse the fill mask
    fill_mask = np.zeros(image.shape[:2]).astype(np.int32)
    for idx, each_mask in enumerate(fill_masks):
        allowed_area = (fill_mask == 0).astype(np.uint8)
        fill_mask += allowed_area * each_mask * categories[idx]

    # collapse the border mask
    border_mask = np.zeros(image.shape[:2]).astype(np.int32)
    for idx, each_mask in enumerate(border_masks):
        allowed_area = (border_mask == 0).astype(np.uint8)
        border_mask += allowed_area * each_mask * categories[idx]

    # embed the fill mask onto image
    mask = visualize_mask(fill_mask, n_classes)
    source = image * np.expand_dims((fill_mask == 0).astype(np.uint8), axis=-1)
    embed_fill = cv2.addWeighted(image, 0.35, mask, 0.65, 0)
    embed_fill *= np.expand_dims((fill_mask != 0).astype(np.uint8), axis=-1)
    embed_fill += source

    # embed the border mask onto embed_fill
    mask = visualize_mask(border_mask, n_classes)
    source = embed_fill * np.expand_dims((border_mask == 0).astype(np.uint8), axis=-1)
    target = cv2.addWeighted(embed_fill, 0.1, mask, 0.9, 0)
    target *= np.expand_dims((border_mask != 0).astype(np.uint8), axis=-1)
    target += source

    return target, fill_mask, border_mask


