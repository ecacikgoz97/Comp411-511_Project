import torch
import torch.nn as nn

class RB(nn.Module):
    """
       Residual BLock For SR without Norm Layer

       conv 1*1
       conv 3*3
       conv 1*1

    """
    def __init__(self, in_channel=64, out_channel=128, in_place=True):
        super(RB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=in_place)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=in_place)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)

        if in_channel != out_channel:
            self.identity = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            def identity(tensor):
                return tensor
            self.identity = identity

    def forward(self, x):

        res = self.conv1(x)
        res = self.relu1(res)
        res = self.conv2(res)
        res = self.relu2(res)
        res = self.conv3(res)
        x = self.identity(x)
        return torch.add(x, res)

class RIB(nn.Module):
    """
        Residual Inception BLock For SR without Norm Layer (for multi-scale processing)
        conv 1*1  conv 1*1  conv 1*1
                  conv 3*3  conv 3*3
                            conv 3*3
                  concat
                  conv 1*1
    """
    def __init__(self, in_channel=64, out_channel=128, activation_inplace=True):
        super(RIB, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1_1 = nn.LeakyReLU(0.2, inplace=activation_inplace)

        self.conv2_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2_2 = nn.LeakyReLU(0.2, inplace=activation_inplace)

        self.conv3_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3_2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3_2 = nn.LeakyReLU(0.2, inplace=activation_inplace)
        self.conv3_3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3_3 = nn.LeakyReLU(0.2, inplace=activation_inplace)

        self.filter_concat = nn.Conv2d(in_channels=3*out_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)

        if in_channel != out_channel:
            self.identity = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            def identity(tensor):
                return tensor
            self.identity = identity

    def forward(self, x):
        res1 = self.conv1_1(x)
        res1 = self.relu1_1(res1)

        res2 = self.conv2_1(x)
        res2 = self.conv2_2(res2)
        res2 = self.relu2_2(res2)

        res3 = self.conv3_1(x)
        res3 = self.conv3_2(res3)
        res3 = self.relu3_2(res3)
        res3 = self.conv3_3(res3)
        res3 = self.relu3_3(res3)

        res = torch.cat((res1, res2, res3), dim=1)
        res = self.filter_concat(res)
        x = self.identity(x)
        out = torch.add(x, res)
        return out


class Encoder(nn.Module):
    """
       Top to Down Block for HourGlass Block
       Consist of ConvNet Block and Pooling
    """
    def __init__(self, in_channel=64, out_channel=64):
        super(Encoder, self).__init__()
        self.res_block = RIB(in_channel=in_channel, out_channel=out_channel)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.res_block(x)
        x2 = self.pool(x)
        return x2, x1

class Decoder(nn.Module):
    """
        Bottom Up Block for HourGlass Block
        Consist of ConvNet Block and Upsampling Block
    """
    def __init__(self, in_channel=64, out_channel=64):
        super(Decoder, self).__init__()
        self.res_block = RIB(in_channel=in_channel, out_channel=out_channel)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, res):
        x = self.upsample(x)
        x = self.res_block(x + res)
        return x

