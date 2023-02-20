import functools

import torchvision.models
import torch.nn.functional as F
from model_tools.check_submission import check_models
import numpy as np
import torch
from torch import nn
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model
from pathlib import Path
import os 

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.


"""
Template module for a base model submission to brain-score
"""




NUM_CLASSES = 1000
img_size = 224
MODEL_NAME = 'lagani-wta-trained-in224'


# from brainscore import score_model
# import functools
# from brainscore import score_model


### UTILITY FUNCTIONS
def get_conv_output_shape(net):
	training = net.training
	net.eval()
	# In order to compute the shape of the output of the network convolutional layers, we can feed the network with
	# a simulated input and return the resulting output shape
	with torch.no_grad(): res = tuple(net.get_conv_output(torch.ones(1, *net.input_shape))[net.CONV_OUTPUT].size())[1:]
	net.train(training)
	return res

def shape2size(shape):
	size = 1
	for s in shape: size *= s
	return size


# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

# define your custom model here:
class Net(nn.Module):
	# Layer names
	CONV1 = 'conv1'
	RELU1 = 'relu1'
	POOL1 = 'pool1'
	BN1 = 'bn1'
	CONV2 = 'conv2'
	RELU2 = 'relu2'
	BN2 = 'bn2'
	CONV3 = 'conv3'
	RELU3 = 'relu3'
	POOL3 = 'pool3'
	BN3 = 'bn3'
	CONV4 = 'conv4'
	RELU4 = 'relu4'
	BN4 = 'bn4'
	CONV_OUTPUT = BN4 # Symbolic name for the last convolutional layer providing extracted features
	FLAT = 'flat'
	FC5 = 'fc5'
	RELU5 = 'relu5'
	BN5 = 'bn5'
	FC6 = 'fc6'
	CLASS_SCORES = FC6 # Symbolic name of the layer providing the class scores as output
	
	def __init__(self, input_shape=(3, img_size, img_size)):
		super(Net, self).__init__()
		
		# Shape of the tensors that we expect to receive as input
		self.input_shape = input_shape
		
		# Here we define the layers of our network
		
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 96, 5) # 3 input channels, 96 output channels, 5x5 convolutions
		self.bn1 = nn.BatchNorm2d(96) # Batch Norm layer
		# Second convolutional layer
		self.conv2 = nn.Conv2d(96, 128, 3) # 96 input channels, 128 output channels, 3x3 convolutions
		self.bn2 = nn.BatchNorm2d(128) # Batch Norm layer
		# Third convolutional layer
		self.conv3 = nn.Conv2d(128, 192, 3)  # 128 input channels, 192 output channels, 3x3 convolutions
		self.bn3 = nn.BatchNorm2d(192) # Batch Norm layer
		# Fourth convolutional layer
		self.conv4 = nn.Conv2d(192, 256, 3)  # 192 input channels, 256 output channels, 3x3 convolutions
		self.bn4 = nn.BatchNorm2d(256) # Batch Norm layer
		
		self.conv_output_size = shape2size(get_conv_output_shape(self))
		
		# FC Layers
		self.fc5 = nn.Linear(self.conv_output_size, 300) # conv_output_size-dimensional input, 300-dimensional output
		self.bn5 = nn.BatchNorm1d(300) # Batch Norm layer
		self.fc6 = nn.Linear(300, NUM_CLASSES) # 300-dimensional input, 10-dimensional output (one per class)
	
	# This function forwards an input through the convolutional layers and computes the resulting output
	def get_conv_output(self, x):
		# Layer 1: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv1_out = self.conv1(x)
		relu1_out = F.relu(conv1_out)
		pool1_out = F.max_pool2d(relu1_out, 2)
		bn1_out = self.bn1(pool1_out)
		
		# Layer 2: Convolutional + ReLU activations + Batch Norm
		conv2_out = self.conv2(bn1_out)
		relu2_out = F.relu(conv2_out)
		bn2_out = self.bn2(relu2_out)
		
		# Layer 3: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv3_out = self.conv3(bn2_out)
		relu3_out = F.relu(conv3_out)
		pool3_out = F.max_pool2d(relu3_out, 2)
		bn3_out = self.bn3(pool3_out)
		
		# Layer 4: Convolutional + ReLU activations + Batch Norm
		conv4_out = self.conv4(bn3_out)
		relu4_out = F.relu(conv4_out)
		bn4_out = self.bn4(relu4_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV1: conv1_out,
			self.RELU1: relu1_out,
			self.POOL1: pool1_out,
			self.BN1: bn1_out,
			self.CONV2: conv2_out,
			self.RELU2: relu2_out,
			self.BN2: bn2_out,
			self.CONV3: conv3_out,
			self.RELU3: relu3_out,
			self.POOL3: pool3_out,
			self.BN3: bn3_out,
			self.CONV4: conv4_out,
			self.RELU4: relu4_out,
			self.BN4: bn4_out
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		# Stretch out the feature map before feeding it to the FC layers
		flat = out[self.CONV_OUTPUT].view(-1, self.conv_output_size)
		
		# Fifth Layer: FC with ReLU activations + Batch Norm
		fc5_out = self.fc5(flat)
		relu5_out = F.relu(fc5_out)
		bn5_out = self.bn5(relu5_out)
		
		# Sixth Layer: dropout + FC, outputs are the class scores
		fc6_out = self.fc6(F.dropout(bn5_out, p=0.5, training=self.training))
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FLAT] = flat
		out[self.FC5] = fc5_out
		out[self.RELU5] = relu5_out
		out[self.BN5] = bn5_out
		out[self.FC6] = fc6_out
		return out



# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_images, image_size=img_size)

mynet = Net()
weights_path = Path(__file__).parent.absolute()
weights_path = os.path.join(weights_path, 'model0.pt')
# print(weights_path)
# print('---')
state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
mynet.load_state_dict(state_dict, strict=False)

# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier=MODEL_NAME, model=mynet, preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier=MODEL_NAME, activations_model=activations_model,
                        # specify layers to consider
                        layers=['conv1', 'conv2', 'conv3', 'conv4', 'fc5']
                        # layers=['conv1', 'relu3', 'relu4']
                        )

# score = score_model(model_identifier=model.identifier, model=model,
#                     benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
# print(score)
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return [MODEL_NAME]


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
    assert name == MODEL_NAME

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = img_size
    return wrapper




# get_layers method to tell the code what layers to consider. If you are submitting a custom
# model, then you will most likley need to change this method's return values.
def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """

    # quick check to make sure the model is the correct one:
    assert name == MODEL_NAME

    # returns the layers you want to consider
    return  ['conv1', 'conv2', 'conv3', 'conv4', 'fc5']


# Bibtex Method. For submitting a custom model, you can either put your own Bibtex if your
# model has been published, or leave the empty return value if there is no publication to refer to.
def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    # from pytorch.py:
    return """custom model from https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb"""

# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
	# pass