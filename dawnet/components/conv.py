import torch
import torch.nn as nn
from dawnet.components.convs import ResidualBasicPreactUnit, ConvBlock, DeconvBlock

# @TODO: just do vanilla forward pass


class UNetEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, concat_last_encoder=False):
        """Initialize the decoder block"""
        inter_channels = out_channels // 2
        self.convs_1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=inter_channels,
                      kernel_size=kernel_size, preact=False, bottleneck=False, padding=1),
            ResidualBasicPreactUnit(in_channels=inter_channels,
                      out_channels=inter_channels, stride=1))

        self.convs_2 = nn.Sequential(
            ConvBlock(in_channels=inter_channels, out_channels=out_channels, kernel_size=3, padding=1),
            ResidualBasicPreactUnit(in_channels=out_channels, out_channels=out_channels, stride=1))

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat_last_encoder = concat_last_encoder
        if self.concat_last_encoder:
            self.transform_1 = ConvBlock(in_channels=2 * inter_channels,
                out_channels=inter_channels, kernel_size=1)
            self.transform_2 = ConvBlock(in_channels=2 * out_channels,
                out_channels=out_channels, kernel_size=1)

    def forward(self, input_tensor, last_down_feature=None):
        """Perform the forward pass"""
        encoder_blocks = {}
        if last_down_feature is not None and not self.concat_last_encoder:
            raise RuntimeError('`concat_last_encoder` should be True')

        hidden = self.convs_1(input_tensor)
        if last_down_feature is not None:
            hidden = torch.cat([last_down_feature[0], hidden], 1)
            hidden = self.transform_1(hidden)
        encoder_blocks[0] = hidden          # @TODO: add self-attention
        hidden = self.max_pool(hidden)
        
        hidden = self.convs_2(input_tensor)
        if last_down_feature is not None:
            hidden = torch.cat([last_down_feature[1], hidden], 1)
            hidden = self.transform_2(hidden)
        encoder_blocks[1] = hidden          # @TODO: add self-attention

        return hidden, encoder_blocks


class UNetDecoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=2):
        inter_channels = in_channels // 2
        self.deconvs_1 = DeConvBlock(
            in_channels=in_channels, out_channels=inter_channels,
            kernel_size=kernel_size, stride=2),
        self.convs = nn.Sequential(
            ResidualBasicPreactUnit(in_channels=out_channels, out_channels=out_channels),
            ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            ResidualBasicPreactUnit(in_channels=out_channels, out_channels=out_channels, stride=1)
        )

    def forward(self, input_tensor, encoder_block):
        """Perform the forward pass"""
        hidden = self.deconvs(input_tensor)
        hidden = torch.cat([hidden, encoder_block], axis=1)
        return self.convs(hidden)



class UNetBlock(nn.Module):
    """Construct the UNET block"""
    
    def __init__(self, in_channels, out_channels, concat_last_encoder):
        """Initialize the object"""
        self.encoder = UNetEncoder(in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, concat_last_encoder=concat_last_encoder)
        self.decoder = UNetDecoder(in_channels=in_channels, out_channels=out_channels, kernel_size=2)

    def forward(self, input_tensor, last_down_feature=None):
        """Perform the forward pass"""
        hidden = self.encoder(input_tensor, last_down_feature)
        output = self.decoder(hidden) 
        

