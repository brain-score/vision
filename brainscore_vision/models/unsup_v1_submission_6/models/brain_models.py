from model_tools.check_submission import check_models
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
import functools
from model_tools.activations.pytorch import PytorchWrapper
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'
from .v1_utils import custom_load_preprocess_images
from brainscore import score_model
from model_tools.brain_transformation import TemporalIgnore, ModelCommitment

class CReLU(nn.Module):
	def __init__(self, inplace=False):
		super(CReLU, self).__init__()

	def forward(self, x):
		x = torch.cat([x,-x],1)
		return F.relu(x)

class L2pooling(nn.Module):
	def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
		super(L2pooling, self).__init__()
		self.padding = (filter_size - 2 )//2
		self.stride = stride
		self.channels = channels
		a = np.hanning(filter_size)[1:-1]
		g = torch.Tensor(a[:,None]*a[None,:])
		g = g/torch.sum(g)
		self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

	def forward(self, input):
		input = input**2
		out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
		return (out+1e-12).sqrt()

class V1Layer(nn.Module):
	def __init__(self, patch_size, kernel_size, num_neurons, nscales=3):
		super().__init__()
		self.patch_size = patch_size
		self.kernel_size = kernel_size
		self.nscales = nscales
		self.hidden = num_neurons
		self.conv_v1 = nn.Conv2d(1, self.hidden, self.kernel_size, stride=1,dilation=1,
									 padding='valid',
									 bias=False)
		#self.conv_v12 = nn.Conv2d(1, self.hidden//2, self.kernel_size,stride=1, dilation=1, padding='valid',bias=False )
		
		
		#self.nonlin = CReLU()
		self.rect = nn.ReLU()
		self.downsample = nn.Upsample(scale_factor=0.5)
		self.pool = L2pooling(3,3,channels=self.hidden)
		
		

		#self.local_onoff = OnOff([5.0],[11.0])

	def forward(self, x, normalize=False):
		
		x = self.conv_v1(x)
		xsimple = self.downsample(self.rect(x))
		xcomplex = self.pool(x)
		x = torch.cat([xsimple, xcomplex], 1)
		
		return x

class V1Brainscore(nn.Module):
	def __init__(self, patch_size, kernel_size, num_neur, normalize=False, device='cuda'):
		super().__init__()

		self.patch_size = patch_size
		self.kernel_size = kernel_size
		self.num_neur = num_neur
		self.normalize = normalize
		self.v1_layer = V1Layer(self.patch_size, self.kernel_size, self.num_neur)
		model_path = os.path.abspath(os.path.join(dir_path, 'V1_model_h230000_21_128.pth'))
		self.v1_layer.load_state_dict(torch.load(model_path, map_location = torch.device(device)))
		self.v1_layer.eval()

	def forward(self, x):
		x = self.v1_layer(x, normalize=self.normalize)
		

		return x

mymodel = V1Brainscore(10, 21, 128, normalize=False, device=device)
mymodel.eval()

# init the model and the preprocessing:
preprocessing = functools.partial(custom_load_preprocess_images, image_size=224, core_object=False)

v1_wrapped = PytorchWrapper(identifier='v1-selfsup8degstr3', model=mymodel, preprocessing=preprocessing)
v1_brain_model = ModelCommitment('v1-selfsup8degstr3',activations_model=v1_wrapped,layers=['v1_layer'],region_layer_map={'V1': 'v1_layer'}, visual_degrees=8)
# get an activations model from the Pytorch Wrapper
#score = score_model(model_identifier='v1-selfsup',model=v1_brain_model,benchmark_identifier= 'movshon.FreemanZiemba2013public.V1-pls')
#print(score)
# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
	"""
	This method defines all submitted model names. It returns a list of model names.
	The name is then used in the get_model method to fetch the actual model instance.
	If the submission contains only one model, return a one item list.
	:return: a list of model string names
	"""

	return ['v1-selfsup8degstr3']


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
	assert name == 'v1-selfsup8degstr3'

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



