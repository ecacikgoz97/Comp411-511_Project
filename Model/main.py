import torch
import torch.nn as nn
from hgsr import HGSR
from blocks_util import calculate_parameters


if __name__ == '__main__':
    model = HGSR(upscale=2, n_HG=3)
    p = calculate_parameters(model)
    print(p)
    model = model.cuda()
    input = torch.FloatTensor(1, 3, 16, 16).cuda()
    result = model(input)
    for tensor in result:
        print(tensor.shape)