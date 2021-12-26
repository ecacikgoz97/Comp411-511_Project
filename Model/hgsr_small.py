import torch.nn as nn
from Model.blocks_model import conv_block, RB, Encoder, Decoder, pixel_shuffle_block
import math


class HG_Block(nn.Module):
    """
    Hour Glass Block for SR Model
    """
    def __init__(self, n_mid=2):
        super(HG_Block, self).__init__()

        self.down_sample1 = Encoder(in_channel=16, out_channel=32)
        self.down_sample2 = Encoder(in_channel=32, out_channel=32)
        self.down_sample3 = Encoder(in_channel=32, out_channel=64)
        self.down_sample4 = Encoder(in_channel=64, out_channel=64)

        res_block = []
        for i in range(n_mid):
            res_block.append(RB(64, 64))
        self.mid_res = nn.Sequential(*res_block)

        self.up_sample1 = Decoder(in_channel=64, out_channel=64)
        self.up_sample2 = Decoder(in_channel=64, out_channel=32)
        self.up_sample3 = Decoder(in_channel=32, out_channel=32)
        self.up_sample4 = Decoder(in_channel=32, out_channel=16)

    def forward(self, x):
        x, res1 = self.down_sample1(x)
        x, res2 = self.down_sample2(x)
        x, res3 = self.down_sample3(x)
        x, res4 = self.down_sample4(x)

        x = self.mid_res(x)

        x = self.up_sample1(x, res4)
        x = self.up_sample2(x, res3)
        x = self.up_sample3(x, res2)
        x = self.up_sample4(x, res1)
        return x

class HGSR(nn.Module):
    """
    Hour Glass SR Model
    """
    def __init__(self, in_channel=3, out_channel=3, upscale=2, nf=16, n_mid=2, n_HG=3):
        super(HGSR, self).__init__()
        self.n_HG = n_HG
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.conv = conv_block(in_channel=in_channel, out_channel=nf, kernel_size=3, padding=1)
        LR_conv = conv_block(in_channel=nf, out_channel=nf, kernel_size=3, padding=1)
        HR_conv1 = conv_block(in_channel=nf, out_channel=nf, kernel_size=3, padding=1)
        HR_conv2 = conv_block(in_channel=nf, out_channel=out_channel, kernel_size=3, padding=1)

        # if upscale == 3:
        #     upsampler = pixel_shuffle_block(in_channel=nf, out_channel=nf)
        # else:
        #     upsampler = [pixel_shuffle_block(in_channel=nf, out_channel=nf) for _ in range(n_upscale)]

        # upsample = nn.Sequential(
        #     LR_conv,
        #     *upsampler,
        #     HR_conv1,
        #     HR_conv2
        # )

        upsample = nn.Sequential(
            LR_conv,
            HR_conv1,
            HR_conv2
        )

        for i in range(n_HG):
            HG = HG_Block(n_mid=n_mid)
            setattr(self, 'HG_%d' % i, HG)
            setattr(self, 'upsample_%d' % i, upsample)


        # # weights initialization
        # for m in self.modules():
        #     if isinstance(m, conv_block):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channel
        #         print(n)
        #         m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.conv(x)
        # results = []
        for i in range(self.n_HG):
            x = getattr(self, 'HG_%d' % i)(x)
            out = getattr(self, 'upsample_%d' % i)(x)
            # results.append(out)

        return out

