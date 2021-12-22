import torch
import torch.nn as nn

def calculate_parameters(model):
    parameters = 0
    for weight in model.parameters():
        p = 1
        for dim in weight.size():
            p *= dim
        parameters += p
    return parameters

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