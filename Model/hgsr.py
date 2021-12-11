import torch.nn as nn
from blocks_model import RB, RIB, Encoder, Decoder
from blocks_util import conv_block, pixel_shuffle_block
import math

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

        self.mid_res1 = RB(in_channel=256, out_channel=256) # Can be replace with RB
        self.mid_res2 = RB(in_channel=256, out_channel=256)

#        self.skip_conv0 = RIB(in_channel=64, out_channel=64)
#        self.skip_conv1 = RIB(in_channel=128, out_channel=128)
#        self.skip_conv2 = RIB(in_channel=128, out_channel=128)
#        self.skip_conv3 = RIB(in_channel=256, out_channel=256)
#        self.skip_conv4 = RIB(in_channel=256, out_channel=256)

        self.up_sample1 = Decoder(in_channel=256, out_channel=256)
        self.up_sample2 = Decoder(in_channel=256, out_channel=128)
        self.up_sample3 = Decoder(in_channel=128, out_channel=128)
        self.up_sample4 = Decoder(in_channel=128, out_channel=64)

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

class HGSR(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, upscale=4, nf=64, n_mid=2, n_HG=3):
        super(HGSR, self).__init__()
        self.n_HG = n_HG
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.conv = conv_block(in_channel=in_channel, out_channel=nf, kernel_size=3)
        LR_conv = conv_block(in_channel=nf, out_channel=nf, kernel_size=3)
        HR_conv1 = conv_block(in_channel=nf, out_channel=nf, kernel_size=3)
        HR_conv2 = conv_block(in_channel=nf, out_channel=out_channel, kernel_size=3)

        if upscale == 3:
            upsampler = pixel_shuffle_block(in_channel=nf, out_channel=nf)
        else:
            upsampler = [pixel_shuffle_block(in_channel=nf, out_channel=nf) for _ in range(n_upscale)]

        upsample = nn.Sequential(
            LR_conv,
            *upsampler,
            HR_conv1,
            HR_conv2
        )

        for i in range(n_HG):
            HG = HG_Block()
            setattr(self, 'HG_%d' % i, HG)
            setattr(self, 'upsample_%d' % i, upsample)

    def forward(self, x):
        x = self.conv(x)
        results = []
        for i in range(self.n_HG):
            x = getattr(self, 'HG_%d' % i)(x)
            out = getattr(self, 'upsample_%d' % i)(x)
            results.append(out)

        return results

