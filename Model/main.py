import torch
import torch.nn as nn
from hgsr_small import HGSR
from simplenet import simpleNet
from blocks_util import calculate_parameters


if __name__ == '__main__':
    model = HGSR(upscale=2, n_HG=3)
    # model = simpleNet(Y=False)
    p = calculate_parameters(model)
    print(p)
    model = model.cuda()
    input = torch.FloatTensor(1, 3, 948, 800).cuda()
    result = model(input)
    for tensor in result:
        print(tensor.shape)