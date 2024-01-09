# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from brainscore_vision.model_helpers.check_submission import check_models
import torch
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

"""
Template module for a base model submission to brain-score
"""

# Bibtex. For submitting a custom model, you can either put your own Bibtex if your
# model has been published, or leave the empty if there is no publication to refer to.
BIBTEX = ""

# LAYERS is list of string layer names to consider per model. The benchmarks maps brain regions to
# layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
# faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
# size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
# model, the layer name are for instance dot concatenated per module, e.g. "features.2".
# If you are submitting a custom model, then you will most likley need to change this variable.
LAYERS = ["conv1", "conv2", "relu1", "relu2", "fully_connected1"]


# Simple CNN for CIFAR10 image classification
class simpleModel(torch.nn.Module):
    def __init__(self):
        super(simpleModel, self).__init__()

        # First convolutional layer:
        # - input channels = 3 - as CIFAR10 contains RGB images)
        # - output channels = 6 - increasing channel depth to improve feature detection
        # - kernel size = 5 - standard kernel size
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.relu1 = torch.nn.ReLU()  # ReLU activation

        # Pooling layer:
        # Identify position and translation invariant features
        # kernel size and stride of 2 - result will be 1/4 of the original number of features
        self.pool = torch.nn.MaxPool2d(2, 2)

        # Second convolutional layer:
        # - input channels = 6 - as the pooling layer does not affect number of channels and this was the output from the first conv layer)
        # - output channels = 12 - again increasing channel depth
        # - kernel size = 5 - standard kernel size
        self.conv2 = torch.nn.Conv2d(6, 12, 5)
        self.relu2 = torch.nn.ReLU()  # ReLU activation

        # First element comes from the batch size (4) multiplied by the output of each kernel (5*5, as kernel size in
        # previous layer is 5), multiplied by the number of channels output from the previous layer (12) (12*5*5*4 = 1200)
        # (This calculation is 'performed' in the flattening operation view() in the forward method)
        self.fully_connected1 = torch.nn.Linear(1200, 120)

        # 10 neurons in output layer to correspond to the 10 categories of object
        self.fully_connected2 = torch.nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(
            x.size(0), -1
        )  # flatten tensor from 12 channels to 1 for the final, linear layers
        x = self.fully_connected1(x)
        x = self.fully_connected2(x)

        return x


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model():
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    # init the model and the preprocessing:
    preprocessing = functools.partial(load_preprocess_images, image_size=32)

    # get an activations model from the Pytorch Wrapper
    wrapper = PytorchWrapper(
        identifier="cb-baseline-model", model=simpleModel(), preprocessing=preprocessing
    )
    wrapper.image_size = 32
    return wrapper


# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
