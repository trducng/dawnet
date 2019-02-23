"""Perform attacks on models.
@author: _john
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from ipywidgets import interact_manual, widgets, Layout
from scipy import stats

from dawnet.data.image import augment_image, resize
from dawnet.data.text import view_string_prediction
from dawnet.diagnose.statistics import normalize_to_range
from dawnet.diagnose.trace import run_partial_model


def changing_input_view_feature_channel(
        image, model, layer_idx=None, channel_idx=None, label=None,
        color_bg=None, construct_widget=True, include=None, preprocess=None):
    """View the feature map as input changes

    Currently this function supports: blur, italicize, noise, rotation,
    padding, perspective transform, elastic transform, brightness. All of them
    are manifested by these keywords:

    ```
    supported_augmentations = [
        'italicize', 'angle', 'pad_vertical', 'pad_horizontal', 'pt',
        'elas_alpha', 'elas_sigma', 'blur', 'brightness', 'gauss_noise',
        'crop'
    ]
    ```

    @NOTE: it should support:
    - random background
    - cropping

    # Argument
        image [2D nd array]: the image
        model [torch.nn.Dawnet]: the dawnet model
        layer_idx [int]: the layer to view. If None, view all layers
        channel_idx [int]: the channel to view. If None, view all channels
        color_bg [int]: the background pixel value (used for interpolation).
            If None, this value will be the mode value of an image
        construct_widget [bool]: whether to construct the widget directly
        include [list of str]: list of augmentation to include, the string
            value is basically the slider arguments. Default is None, which
            means all
        preprocess [func]: a preprocess function passed to `run_partial_model`
    """
    if color_bg is None:
        color_bg = int(stats.mode(image+128, axis=None).mode.item())
    if next(model.parameters()).is_cuda:
        image_torch = torch.FloatTensor(image, device=torch.device('cuda:0'))
        image_torch = image_torch.unsqueeze(0).unsqueeze(0).cuda()
    else:
        image_torch = torch.FloatTensor(image)
        image_torch = image_torch.unsqueeze(0).unsqueeze(0).cuda()

    supported_augmentations = [
        'italicize', 'angle', 'pad_vertical', 'pad_horizontal', 'pt',
        'elas_alpha', 'elas_sigma', 'blur', 'brightness', 'gauss_noise',
        'crop'
    ]
    include = supported_augmentations if include is None else include
    if set(include).difference(set(supported_augmentations)):
        raise AttributeError(
            'unknown argumentation types, support {} but receive {}'.format(
                supported_augmentations,
                set(include).difference(set(supported_augmentations))
            ))

    fig = plt.figure()

    def activate(aug_type):
        """Activate the augmentation"""
        if aug_type in include:
            return Layout(width='75%')
        else:
            return Layout(display='none')

    def show_images(layer, page, italicize, angle, pad_vertical,
                    pad_horizontal, pt, elas_alpha, elas_sigma, blur_type,
                    blur_value, brightness, gauss_noise, crop):

        fig.clf()

        # enhance the image
        top, bottom, left, right = crop.replace(' ', '').split(',')
        image_new = image[int(top):int(bottom), int(left):int(right)]
        image_new = augment_image(
            image=image_new, color_bg=color_bg,
            italicize=italicize, angle=angle, pad_vertical=pad_vertical,
            pad_horizontal=pad_horizontal, pt=pt, elas_alpha=elas_alpha,
            elas_sigma=elas_sigma, blur_type=blur_type, blur_value=blur_value,
            brightness=brightness, gauss_noise=gauss_noise)
        image_new = resize(image_new, height=64)
        image_new_infer = image_new.astype(np.float32)
        prediction = model.x_infer(image_new_infer)

        if label is None:
            print('Prediction: {}'.format(prediction))
        else:
            print('Ground truth: {} - Prediction: '.format(label), end='')
            _ = view_string_prediction(prediction, label, to_print=True,
                                       notebook=True)

        if next(model.parameters()).is_cuda:
            image_torch = torch.FloatTensor(
                image_new, device=torch.device('cuda:0'))
            image_torch = image_torch.unsqueeze(0).unsqueeze(0).cuda()
        else:
            image_torch = torch.FloatTensor(image_new)
            image_torch = image_torch.unsqueeze(0).unsqueeze(0).cuda()

        # retrieve the feature map
        layer = layer if layer_idx is None else layer_idx
        feature_map = run_partial_model(model, layer, image_torch,
                                        preprocess=preprocess)
        feature_map = feature_map.squeeze().cpu().data.numpy()

        # case where viewing all channels in a layer
        if channel_idx is None:
            image_list = list(feature_map[page*15:(page+1)*15])
            columns = min(3, len(image_list))
            rows = math.ceil(len(image_list) / columns)

            for _idx, each_img in enumerate(image_list):
                plot = fig.add_subplot(rows+1, columns, _idx+1)
                plot.imshow(each_img, cmap='gray')

            plot = fig.add_subplot(rows+1, columns, ((rows+1) *columns)-1)
            plot.imshow(image_new, cmap='gray')

        # case where viewing only 1 channel in a layer
        else:
            plot = fig.add_subplot('211')
            plot.imshow(image_new, cmap='gray')
            plot = fig.add_subplot('212')
            plot.imshow(feature_map[channel_idx], cmap='gray')

        plt.subplots_adjust(
            left=0, right=1, top=0.9, bottom=0, wspace=0.01, hspace=0.01)
        fig.show()

    if not construct_widget:
        return show_images

    def update_blur_value(*args):
        """Update the blur range based on the blur type"""
        if blur_type.value == 1:
            blur_value.max = 15
            blur_value.min = 0
            blur_value.step = 1
            blur_value.value = 0
        elif blur_type.value == 2 or blur_type.value == 3:
            blur_value.max = 7
            blur_value.min = 1
            blur_value.step = 2
            blur_value.value = 1

    def update_num_channels(*args):
        """Change the number of channels as the layer changes"""
        feature_map = run_partial_model(model, layer_slider.value, image_torch)
        max_page = feature_map.shape[1] // 15
        page_slider.max = max_page

    # sliders containing blur
    blur_type = widgets.Dropdown(
        options=[('N/A', 0), ('Gaussian', 1), ('Average', 2), ('Median', 3)],
        value=0, description='Blur type:', layout=activate('blur'))
    blur_value = widgets.IntSlider(
        min=0, max=10, step=1, value=0, description='Blur value:',
        layout=activate('blur'))
    blur_type.observe(update_blur_value, 'value')

    # layer/channel relating-slider
    if layer_idx is not None:
        layer_slider = widgets.IntSlider(min=0, max=10, step=1, value=0,
                                         layout=Layout(display='none'))
        if channel_idx is None:
            feature_map = run_partial_model(model, layer_idx, image_torch,
                                            preprocess=preprocess)
            max_page = feature_map.shape[1] // 15
            page_slider = widgets.IntSlider(
                min=0, max=max_page, step=1, value=0,
                description='Channels:', layout=Layout(width='75%'))
        else:
            page_slider = widgets.IntSlider(
                min=0, max=10, step=1, value=0,
                description='Channels:', layout=Layout(display='none'))
    else:
        layer_slider = widgets.IntSlider(
            min=0, max=model.get_number_layers(), step=1, value=0,
            layout=Layout(width='75%'))
        if channel_idx is None:
            feature_map = run_partial_model(model, 0, image_torch,
                                            preprocess=preprocess)
            max_page = feature_map.shape[1] // 15
            page_slider = widgets.IntSlider(
                min=0, max=max_page, step=1, value=0,
                description='Channels:', layout=Layout(width='75%'))
            layer_slider.observe(update_num_channels, 'value')
        else:
            page_slider = widgets.IntSlider(
                min=0, max=10, step=1, value=0, description='Channels:',
                layout=Layout(display='none'))

    # interactive sliders
    interact_manual(
        show_images,
        layer=layer_slider,
        page=page_slider,
        italicize=widgets.FloatSlider(
            min=-30, max=30, step=0.5, value=0, description='Italicize:',
            orientation='horizontal', layout=activate('italicize')),
        angle=widgets.FloatSlider(
            min=-10, max=10, step=0.5, value=0, description='Rotate:',
            layout=activate('angle')),
        pad_vertical=widgets.FloatSlider(
            min=0, max=0.7, step=0.01, value=0, description='Pad (vertical):',
            layout=activate('pad_vertical')),
        pad_horizontal=widgets.FloatSlider(
            min=0, max=0.7, step=0.01, value=0,
            description='Pad (horizontal):',
            layout=activate('pad_horizontal')),
        pt=widgets.FloatSlider(
            min=0, max=0.3, step=0.02, value=0, description='Perspective:',
            layout=activate('pt')),
        elas_alpha=widgets.FloatSlider(
            min=0, max=1.0, step=0.05, value=0, description='Elastic (alpha):',
            layout=activate('elas_alpha')),
        elas_sigma=widgets.FloatSlider(
            min=0.4, max=0.6, step=0.05, value=0.4,
            description='Elastic (sigma)', layout=activate('elas_sigma')),
        blur_type=blur_type,
        blur_value=blur_value,
        brightness=widgets.FloatSlider(
            min=0.3, max=1.8, step=0.1, value=1, description='Brightness:',
            layout=activate('brightness')),
        gauss_noise=widgets.FloatSlider(
            min=0, max=0.2, step=0.01, value=0, description='Gauss noise:',
            layout=activate('gauss_noise')),
        crop=widgets.Text(
            value='0,{},0,{}'.format(*image.shape),
            placeholder='0,{},0,{}'.format(*image.shape),
            description='Crop (top,bototm,left,right):',
            layout=activate('crop')))

    return show_images


def compare_model_response(
        model, input1, input2, layer_idx=None, channel_idx=None,
        construct_widget=True, preprocess=None):
    """Compare a model response, given 2 inputs

    # Arguments
        model [torch.nn.Module]: the dawnet model
        input1 [2D nd array]: a valid input to the model
        input2 [2D nd array]: a valid input to the model
        layer_idx [int]: the layer to view. If None, view all layers
        channel_idx [int]: the channel to view. If None, view all channels
        construct_widget [bool]: whether to construct the widget directly
        preprocess [func]: a preprocessing function applied to the input

    # Returns
        [func]: image showing function for ipywidget interactivity
    """

    # process the input to torch data
    if next(model.parameters()).is_cuda:
        input1_torch = torch.FloatTensor(
            input1, device=torch.device('cuda:0'))
        input1_torch = input1_torch.unsqueeze(0).unsqueeze(0).cuda()
        input2_torch = torch.FloatTensor(
            input2, device=torch.device('cuda:0'))
        input2_torch = input2_torch.unsqueeze(0).unsqueeze(0).cuda()
    else:
        input1_torch = torch.FloatTensor(input1)
        input1_torch = input1_torch.unsqueeze(0).unsqueeze(0).cuda()
        input2_torch = torch.FloatTensor(input2)
        input2_torch = input2_torch.unsqueeze(0).unsqueeze(0).cuda()

    fig = plt.figure()

    def show_images(layer, page, crop):
        """Show the layers interactively"""

        left, right = crop.replace(' ', '').split(',')
        left, right = int(left), int(right)
        pred1 = model.x_infer(input1)
        pred2 = model.x_infer(input2)
        print('{} - {}'.format(pred1, pred2))

        # retrieve the feature map
        layer = layer if layer_idx is None else layer_idx
        feature1_map = run_partial_model(model, layer, input1_torch,
                                         preprocess=preprocess)
        feature1_map = feature1_map.squeeze().cpu().data.numpy()
        feature1_map = feature1_map[:, :, left:right]
        feature2_map = run_partial_model(model, layer, input2_torch,
                                         preprocess=preprocess)
        feature2_map = feature2_map.squeeze().cpu().data.numpy()
        feature2_map = feature2_map[:, :, left:right]

        # case where viewing all channels in a layer
        if channel_idx is None:
            image1_list = list(feature1_map[page*5:(page+1)*5])
            image2_list = list(feature2_map[page*5:(page+1)*5])
            image_diff = (feature1_map[page*5:(page+1)*5] -
                          feature2_map[page*5:(page+1)*5])
            diff_min, diff_max = np.min(image_diff), np.max(image_diff)
            image_diff_list = list(normalize_to_range(
                image_diff, min_value=0, max_value=255,
                current_range=(diff_min, diff_max)))
            columns = 3
            rows = min(5, len(image1_list))

            for _idx in range(rows):
                image1 = image1_list[_idx]
                image2 = image2_list[_idx]
                image_diff = image_diff_list[_idx]

                min_value = min(np.min(image1), np.min(image2))
                max_value = max(np.max(image1), np.max(image2))

                image1 = normalize_to_range(
                    image1, min_value=0, max_value=255,
                    current_range=(min_value, max_value))
                image2 = normalize_to_range(
                    image2, min_value=0, max_value=255,
                    current_range=(min_value, max_value))

                plot = fig.add_subplot(rows+1, columns, _idx*3+1)
                plot.imshow(image1, cmap='gray')
                plot.axis('off')
                plot = fig.add_subplot(rows+1, columns, _idx*3+2)
                plot.imshow(image2, cmap='gray')
                plot.set_title('{}'.format(_idx + page*5))
                plot.axis('off')
                plot = fig.add_subplot(rows+1, columns, _idx*3+3)
                plot.imshow(image_diff, cmap='gray', vmin=0, vmax=255)
                plot.axis('off')

        # case where viewing only 1 channel in a layer
        else:
            image1 = feature1_map[channel_idx]
            image2 = feature2_map[channel_idx]
            image_diff = normalize_to_range(
                image1 - image2, min_value=0, max_value=255)

            min_value = min(np.min(image1), np.min(image2))
            max_value = max(np.max(image1), np.max(image2))

            image1 = normalize_to_range(
                image1, min_value=0, max_value=255,
                current_range=(min_value, max_value))
            image2 = normalize_to_range(
                image2, min_value=0, max_value=255,
                current_range=(min_value, max_value))

            plot = fig.add_subplot('131')
            plot.imshow(image1, cmap='gray')
            plot.axis('off')
            plot = fig.add_subplot('132')
            plot.imshow(image2, cmap='gray')
            plot.axis('off')
            plot = fig.add_subplot('133')
            plot.imshow(image_diff, cmap='gray', vmin=0, vmax=255)
            plot.axis('off')

        plt.subplots_adjust(
            left=0, right=1, top=0.9, bottom=0, wspace=0.01, hspace=0.01)
        fig.show()

    if not construct_widget:
        return show_images

    def layer_update(*args):
        """Change the number of channels as the layer changes"""
        feature_map = run_partial_model(
            model, layer_slider.value, input1_torch)
        max_page = feature_map.shape[1] // 5
        page_slider.max = max_page

        crop_widget.value = '0,{}'.format(feature_map.shape[-1])
        crop_widget.placeholder = '0,{}'.format(feature_map.shape[-1])

    # layer/channel relating-slider and crop widget
    if layer_idx is not None:
        layer_slider = widgets.IntSlider(min=0, max=10, step=1, value=0,
                                         layout=Layout(display='none'))
        feature_map = run_partial_model(model, layer_idx, input1_torch,
                                        preprocess=preprocess)
        crop_widget = widgets.Text(
            value='0,{}'.format(feature_map.shape[-1]),
            placeholder='0,{}'.format(feature_map.shape[-1]),
            description='Crop (left,right):',
            layout=Layout(width='75%'))

        if channel_idx is None:
            max_page = feature_map.shape[1] // 5
            page_slider = widgets.IntSlider(
                min=0, max=max_page, step=1, value=0,
                description='Channels:', layout=Layout(width='75%'))
        else:
            page_slider = widgets.IntSlider(
                min=0, max=10, step=1, value=0,
                description='Channels:', layout=Layout(display='none'))
    else:
        layer_slider = widgets.IntSlider(
            min=0, max=model.get_number_layers(), step=1, value=0,
            layout=Layout(width='75%'))
        feature_map = run_partial_model(model, 0, input1_torch,
                                        preprocess=preprocess)
        crop_widget = widgets.Text(
            value='0,{}'.format(feature_map.shape[-1]),
            placeholder='0,{}'.format(feature_map.shape[-1]),
            description='Crop (left,right):',
            layout=Layout(width='75%'))

        if channel_idx is None:
            max_page = feature_map.shape[1] // 5
            page_slider = widgets.IntSlider(
                min=0, max=max_page, step=1, value=0,
                description='Channels:', layout=Layout(width='75%'))
            layer_slider.observe(layer_update, 'value')
        else:
            page_slider = widgets.IntSlider(
                min=0, max=10, step=1, value=0, description='Channels:',
                layout=Layout(display='none'))

    interact_manual(
        show_images,
        layer=layer_slider,
        page=page_slider,
        crop=crop_widget)

    return show_images
