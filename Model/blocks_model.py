import torch
import torch.nn as nn

def conv_block(in_channel, out_channel, kernel_size, stride=1, padding=0, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
        nn.BatchNorm2d(out_channel, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class RB(nn.Module):
    def __init__(self, in_channel=64, out_channel=128, in_place=True):
        super(RB, self).__init__()
        conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=in_place)
        conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=in_place)
        conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.res = nn.Sequential(conv1, leaky_relu1, conv2, leaky_relu2, conv3)
        if in_channel != out_channel:
            self.identity = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            def identity(tensor):
                return tensor
            self.identity = identity

    def forward(self, x):
        res = self.res(x)
        x = self.identity(x)
        return torch.add(x, res)


class Encoder(nn.Module):
    """
       Top to Down Block for HourGlass Block
       Consist of ConvNet Block and Pooling
    """
    def __init__(self, in_channel=64, out_channel=64):
        super(Encoder, self).__init__()
        self.res_block = RB(in_channel=in_channel, out_channel=out_channel)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.res_block(x)
        return self.max_pool(x), x

class Decoder(nn.Module):
    """
        Bottom Up Block for HourGlass Block
        Consist of ConvNet Block and Upsampling Block
    """
    def __init__(self, in_channel=64, out_channel=64):
        super(Decoder, self).__init__()
        self.res_block = RB(in_channel=in_channel, out_channel=out_channel)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x, res):
        x = self.upsample(x)
        return self.res_block(x + res)


def pixel_shuffle_block(in_channel, out_channel, upscale_factor=2, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        conv_block(in_channel=in_channel, out_channel=out_channel * (upscale_factor ** 2), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.PixelShuffle(upscale_factor=upscale_factor),
        nn.BatchNorm2d(out_channel, affine=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )