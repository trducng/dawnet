from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(modules):
    """Initialize module weights"""
    pass


class UNetBlock(nn.Module):
    """A UNet block

    # Arguments
        in_channels [int]: the number of input channels to the unet block
    """

    def __init__(self, in_channels):
        """Initialize the object"""
        super(UNetBlock, self).__init__()
        self.in_channels = in_channels

        self.block_1 = ResBlock(in_channels=in_channels, out_channels=in_channels*2)
        self.block_1_trans = ConvBlock(
            in_channels=in_channels*4, out_channels=in_channels*2)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block_2 = ResBlock(in_channels=in_channels*2, out_channels=in_channels*4)
        self.block_2_trans = ConvBlock(
            in_channels=in_channels*8, out_channels=in_channels*4)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block_3 = ResBlock(in_channels=in_channels*4, out_channels=in_channels*8)
        self.block_3_trans = ConvBlock(
            in_channels=in_channels*16, out_channels=in_channels*8)

        self.block_trans = ResBlock(
            in_channels=in_channels*8, out_channels=in_channels*8)
        self.up_block_1 = ResBlock(
            in_channels=in_channels*16, out_channels=in_channels*4)
        self.up_block_2 = ResBlock(
            in_channels=in_channels*8, out_channels=in_channels*2)
        self.up_block_3 = ResBlock(
            in_channels=in_channels*4, out_channels=in_channels)

    def forward(self, x, last_encoder_features=None):
        """Perform the forward pass

        # Arguments
            x [4D array]: input feature of shape B x C x H x W
            last_encoder_features [tuple of 4D arrays]: the intermediate outputs of
                last UNet's encoder, each 4D array has shape B x C x H x W

        # Returns
            [4D array]: the output feature of shape B x C x H x W
            [tuple of 4D arrays]: the intermediate output of each UNet's encoder
        """
        # Downsampling phase
        h = self.block_1(x)
        if last_encoder_features is not None:
            h = torch.cat([h, last_encoder_features[0]], dim=1)
            h = self.block_1_trans(h)
        h_1 = h
        h = self.max_pool_1(h)

        h = self.block_2(h)
        if last_encoder_features is not None:
            h = torch.cat([h, last_encoder_features[1]], dim=1)
            h = self.block_2_trans(h)
        h_2 = h
        h = self.max_pool_2(h)

        h = self.block_3(h)
        if last_encoder_features is not None:
            h = torch.cat([h, last_encoder_features[2]], di=1)
            h = self.block_3_trans(h)
        h_3 = h

        EncoderFeatures = namedtuple('EncoderFeatures', ['h1, h2, h3'])
        encoder_features = EncoderFeatures(h_1, h_2, h_3)

        h = self.block_trans(h)

        # Upsampling phase
        h = torch.cat([h, h_3], dim=1)
        h = self.up_block_1(h)

        h = F.interpolate(h, size=h_2.size()[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, h_2], dim=1)
        h = self.up_block_2(h)

        h = F.interpolate(h, size=h_1.size()[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, h_1], dim=1)
        h = self.up_block_3(h)

        return h, encoder_features


class ConvBlock(nn.Module):
    """Default convolution block employed in MSAU"""

    def __init__(self, in_channels, out_channels, activation=nn.ReLU(),
                 dilation=1,  drop_rate=0.0):
        """Initialize the object"""
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3,
            dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.lrn = nn.LocalResponseNorm(size=out_channels)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        """Perform the forward pass"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrn(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout(x)

        return x


class ResBlock(nn.Module):
    """Residual block"""

    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        """Initialize the object"""
        super(ResBlock, self).__init__()
        self.convs = ConvBlock(in_channels, out_channels, drop_rate=drop_rate),
        self.res_convs = nn.Sequential(
            ConvBlock(out_channels, out_channels, drop_rate=drop_rate),
            ConvBlock(out_channels, out_channels, activation=None, drop_rate=drop_rate))

    def forward(self, x):
        """Perform the forward pass"""
        h = self.convs(x)

        return F.relu(x + h)




