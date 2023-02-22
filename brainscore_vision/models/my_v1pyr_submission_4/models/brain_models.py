
from model_tools.check_submission import check_models
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import functools
from model_tools.activations.pytorch import PytorchWrapper
from .v1_utils import custom_load_preprocess_images
from brainscore import score_model
from model_tools.brain_transformation import TemporalIgnore, ModelCommitment
from .steerable_pyramid_freq import Steerable_Pyramid_Freq

class CReLU(nn.Module):
	def __init__(self, inplace=False):
		super(CReLU, self).__init__()

	def forward(self, x):
		#x = rearrange([x, -x], 't b c h w -> b (c t) h w')
		x = torch.cat([x,-x],1)
		return F.relu(x)

class V1Net(nn.Module):
	#V1 layer implemented using steerable frequency domain pyramid
	def __init__(self, imshape, order, scales, include_list = ['simple','complex']
				 ,eval = False, is_complex = True, downsample=False, tight_frame=False):
		super(V1Net, self).__init__()

		self.order = order
		self.scales = scales
		self.imshape = imshape
		self.include_list = include_list
		self.is_complex = is_complex
		self.downsample = downsample
		self.tight_frame = tight_frame
		self.cell_dict = {'simple': None, 'complex': None}
		self.pyr = Steerable_Pyramid_Freq(image_shape = self.imshape, height=self.scales, order = self.order,
																	is_complex=self.is_complex, downsample=self.downsample, tight_frame=self.tight_frame)
		if self.scales == 'auto':
			self.scales = self.pyr.num_scales
		self.rectify = nn.ReLU(inplace=True)
	
	def rearrange_coeffs(self,pyr_coeffs, exclude=[], max_size =32 ):
			tensor_dict = {}
			for channel in pyr_coeffs.keys():				
				if channel not in exclude:
					if isinstance(channel, tuple):
						scale = channel[0]
						if scale not in exclude:
							#if scale == 0:
							out = pyr_coeffs[channel]
							mid = out.shape[2]//2
							start = max(0,int(mid - max_size//2))
							end = int(mid + max_size//2)
							if end > out.shape[2]-1:
								end = out.shape[2]-1
							#start = 0
							#end = out.shape[2]-1
							if scale not in tensor_dict.keys():
								tensor_dict[scale] = pyr_coeffs[channel][...,start:end,start:end]
							else:
								tensor_dict[scale] = torch.cat([tensor_dict[scale], pyr_coeffs[channel][...,start:end,start:end]],dim=1)
						
					
					else:
						tensor_dict[channel] = pyr_coeffs[channel]

			return tensor_dict

	def to(self, *args, **kwargs):
		self.pyr = self.pyr.to(*args, **kwargs)
		return super().to(*args, **kwargs)

	def flatten(self, channel):
		channel = channel.view(channel.shape[0], -1)

		return channel

	def forward(self, x, exclude=[]):
		batch_size = x.shape[0]
		coeffs = self.pyr.forward(x,[0,1,2,3,4,5])
		tensor_dict = self.rearrange_coeffs(coeffs, exclude)
		for scale in tensor_dict.keys():
			if tensor_dict[scale].shape[1] > 1:
				simple_cells = tensor_dict[scale]
				if 'complex' in self.include_list:
					complex_cells = (simple_cells.real.square() + simple_cells.imag.square()).sqrt()
				if 'simple' in self.include_list:
					simple_cells = torch.cat((simple_cells.real, simple_cells.imag), dim=1)
					simple_cells = self.rectify(simple_cells)
				if ('simple' in self.include_list) and ('complex' in self.include_list):
					tensor_dict[scale] = torch.cat([simple_cells, complex_cells], dim=1)
				elif 'simple' in self.include_list:
					tensor_dict[scale] = simple_cells
				else:
					tensor_dict[scale] = complex_cells


			tensor_dict[scale] = self.flatten(tensor_dict[scale])
		
		return torch.cat(list(tensor_dict.values()), 1)



"""
Template module for a base model submission to brain-score
"""
class V1PyrBrainscore(nn.Module):
	def __init__(self, image_size, num_ori, num_scales, downsample=True, exclude=[]):
		super(V1PyrBrainscore, self).__init__()
		self.v1_layer = V1Net(image_size, num_ori, num_scales, downsample=downsample, include_list=['simple','complex'], tight_frame=True)
		self.exclude = exclude
	def forward(self,x):
		self.v1_layer = self.v1_layer.to(x.device)
		out = self.v1_layer(x, self.exclude)

		return out



# init the model and the preprocessing:
mymodel = V1PyrBrainscore([256,256], 3, 'auto', downsample=True, exclude=['residual_highpass', 'residual_lowpass'])
preprocessing = functools.partial(custom_load_preprocess_images, image_size=256, core_object=False)

v1_wrapped = PytorchWrapper(identifier='v1-pyr-sel32', model=mymodel, preprocessing=preprocessing)
v1_brain_model = ModelCommitment('v1-pyr-sel32',activations_model=v1_wrapped,layers='v1_layer', region_layer_map={'V1':'v1_layer'}, visual_degrees=8)
# get an activations model from the Pytorch Wrapper


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
	"""
	This method defines all submitted model names. It returns a list of model names.
	The name is then used in the get_model method to fetch the actual model instance.
	If the submission contains only one model, return a one item list.
	:return: a list of model string names
	"""

	return ['v1-pyr-sel32']


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model(name):
	"""
	This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
	containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
	keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
	wrappers.
	:param name: the name of the model to fetch
	:return: the model instance
	"""
	assert name == 'v1-pyr-sel32'

	# link the custom model to the wrapper object(activations_model above):
	model = v1_brain_model
	return v1_brain_model



# Bibtex Method. For submitting a custom model, you can either put your own Bibtex if your
# model has been published, or leave the empty return value if there is no publication to refer to.
def get_bibtex(model_identifier):
	"""
	A method returning the bibtex reference of the requested model as a string.
	"""

	# from pytorch.py:
	return ''

# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
	# Use this method to ensure the correctness of the BaseModel implementations.
	# It executes a mock run of brain-score benchmarks.
	check_models.check_brain_models(__name__)

