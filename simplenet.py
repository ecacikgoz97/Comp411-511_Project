import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable

class simpleNet(nn.Module):
	def __init__(self,Y=False):
		super(simpleNet, self).__init__()
		d = 1
		if Y == False:
			d = 3
		self.input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1,   bias=True)
		self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
		self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
		self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
		self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
		self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
		self.up1 = nn.PixelShuffle(2)
		self.up2 = nn.PixelShuffle(2)
	
		self.output = nn.Conv2d(in_channels=8, out_channels=d, kernel_size=3, stride=1, padding=1, bias=True)
		self.relu = nn.ReLU(inplace=True)
        


		# weights initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, sqrt(2. / n))

	def forward(self, x):
		residual = x
		inputs = self.input(self.relu(x))
		out = inputs
		
		out = self.conv1(self.relu(out))
		out = self.conv2(self.relu(out))
		out = self.conv3(self.relu(out))
		out = self.conv4(self.relu(out))
		out = self.conv5(self.relu(out))
		out = self.conv6(self.relu(out))
		out = self.up2(self.up1(out))

		#out = torch.add(out, inputs)

		out = self.output(self.relu(out))
		
# 		out = torch.add(out, residual)
		return out
