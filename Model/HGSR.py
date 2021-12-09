import torch.nn as nn
from blocks_model import RB, RIB, Encoder, Decoder

class HG_Block(nn.Module):
    """
    Hour Glass Block for SR Model
    """
    def __init__(self):
        super(HG_Block, self).__init__()

        self.down_sample1 = Encoder(in_channel=64, out_channel=128)
        self.down_sample2 = Encoder(in_channel=128, out_channel=128)
        self.down_sample3 = Encoder(in_channel=128, out_channel=256)
        self.down_sample4 = Encoder(in_channel=256, out_channel=256)

        self.mid_res1 = RIB(in_channel=256, out_channel=256) # Can be replace with RB
        self.mid_res2 = RIB(in_channel=256, out_channel=256)

#        self.skip_conv0 = RIB(in_channel=64, out_channel=64)
#        self.skip_conv1 = RIB(in_channel=128, out_channel=128)
#        self.skip_conv2 = RIB(in_channel=128, out_channel=128)
#        self.skip_conv3 = RIB(in_channel=256, out_channel=256)
#        self.skip_conv4 = RIB(in_channel=256, out_channel=256)

        self.up_sample1 = Decoder(in_channel=256, out_channel=256)
        self.up_sample1 = Decoder(in_channel=256, out_channel=128)
        self.up_sample1 = Decoder(in_channel=128, out_channel=128)
        self.up_sample1 = Decoder(in_channel=128, out_channel=64)

    def forward(self, x):
        x, res1 = self.down_sample1(x)
        x, res2 = self.down_sample2(x)
        x, res3 = self.down_sample3(x)
        x, res4 = self.down_sample4(x)

        x = self.mid_res1(x)
        x = self.mid_res2(x)

        x = self.up_sample1(x, res4)
        x = self.up_sample2(x, res3)
        x = self.up_sample3(x, res2)
        x = self.up_sample4(x, res1)
        return x
