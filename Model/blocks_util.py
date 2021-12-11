import torch
import torch.nn as nn

class ConcatBlock(nn.Module):
    """
    Concatenate the output of a submodule to its input.
    """
    def __init__(self, sub_module):
        super(ConcatBlock, self).__init__()
        self.submodule = sub_module

    def forward(self, x):
        output = torch.cat((x, self.submodule(x)), dim=1)
        return output


def conv_block(in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
        nn.BatchNorm2d(out_channel, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )

def pixel_shuffle_block(in_channel, out_channel, upscale_factor=2, kernel_size=3, stride=1, padding=0, bias=True):
    """
        Pixel Shuffle Layer
    """
    return nn.Sequential(
        conv_block(in_channel=in_channel, out_channel=out_channel * (upscale_factor ** 2), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.PixelShuffle(upscale_factor=upscale_factor),
        nn.BatchNorm2d(out_channel, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )

def calculate_parameters(model):
    parameters = 0
    for weight in model.parameters():
        p = 1
        for dim in weight.size():
            p *= dim
        parameters += p
    return parameters

