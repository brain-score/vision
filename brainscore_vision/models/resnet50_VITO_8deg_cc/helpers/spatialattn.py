import torch
from torch import nn

class DSConv2d(nn.Module):
	def __init__(self, nin, kernels_per_layer, nout): 
		super(DSConv2d, self).__init__() 
		self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin, padding_mode='reflect') 
		self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1) 
  
	def forward(self, x): 
		out = self.depthwise(x) 
		out = self.pointwise(out) 
		return out


class SpatialAttn(nn.Module):
	def __init__(self, inp_dim, hidden_sizes, num_heads, atttype='channel'):
		super(SpatialAttn, self).__init__() 
		self.inp_dim = inp_dim
		self.hidden_sizes = hidden_sizes
		self.num_heads = num_heads
		
		self.module_list = nn.ModuleList([])
		if atttype == 'channel':
			first_pool = nn.Conv2d(self.inp_dim, hidden_sizes[0], kernel_size=1, bias=True)
		elif atttype == 'dw':
			first_pool = DSConv2d(self.inp_dim, 1, hidden_sizes[0])
		for i in range(len(hidden_sizes)+1):
			if i == 0:
				self.module_list.append(first_pool)
				self.module_list.append(nn.ReLU(inplace=True))
			elif i < len(hidden_sizes):
				self.module_list.append(nn.Conv2d(hidden_sizes[i-1], hidden_sizes[i], kernel_size=1, bias=True))
				self.module_list.append(nn.ReLU(inplace=True))
			else:
				self.module_list.append(nn.Conv2d(hidden_sizes[i-1], num_heads, kernel_size=1, bias=False))
		

	def forward(self, x):
		inp=x
		for m in self.module_list:
			x = m(x)
		assert self.inp_dim % x.shape[1] == 0, "Output channels must be divisible by input number of channels"
		x = nn.Softmax(dim=-1)(x.view(x.shape[0], x.shape[1], -1)).view(x.shape)
		if x.shape[1] != self.inp_dim:
			x = x.repeat(1, self.inp_dim // x.shape[1], 1, 1)
		w = x
		x = w*inp
		x = x.sum((-1,-2))
		return x
