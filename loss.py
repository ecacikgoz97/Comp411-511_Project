import torch
import torch.nn as nn
import torch.nn.functional as F

def L1_loss(x1, x2, mask=1):
    return torch.mean(torch.abs(x1 - x2) * mask)

def L2_loss(x1, x2, mask=1):
    return torch.mean(((x1-x2) * mask) ** 2)

def C_loss(x1, x2):
    """L1 Charbonnierloss."""
    diff = torch.add(x1, -x2)
    error = torch.sqrt(diff * diff + 1e-6)
    loss = torch.sum(error)
    return loss

def get_content_loss(loss_type, nn_func=True, use_cuda=False):
    """
    content loss: [l1, l2, c]
    """

    if loss_type == 'l2':
        loss = nn.MSELoss() if nn_func else L2_loss
    elif loss_type == 'l1':
        loss = nn.L1Loss() if nn_func else L1_loss
    elif loss_type == 'c':
        loss = C_loss
    else:
        loss = nn.MSELoss() if nn_func else L2_loss
    if use_cuda and nn_func:
        return loss.cuda()
    else:
        return loss

def GW_loss(x1, x2):
    sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
    b, c, w, h = x1.shape
    sobel_x = torch.FloatTensor(sobel_x).expand(c, 1, 3, 3)
    sobel_y = torch.FloatTensor(sobel_y).expand(c, 1, 3, 3)
    sobel_x = sobel_x.type_as(x1)
    sobel_y = sobel_y.type_as(x1)
    weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
    weight_y = nn.Parameter(data=sobel_y, requires_grad=False)
    Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
    Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
    Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
    Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
    dx = torch.abs(Ix1 - Ix2)
    dy = torch.abs(Iy1 - Iy2)
#     loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
    loss = (1 + 4*dx) * (1 + 4*dy) * torch.abs(x1 - x2)
    return torch.mean(loss)



