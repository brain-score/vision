from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
import os


class YassineTest1(torch.nn.Module):

    def __init__(self):
        super(YassineTest1, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def get_model_list():
    return ["Yassine_test_1"]


MODEL = YassineTest1()

def get_layers(name):
    assert name == "Yassine_test_1"
    layer_names = []

    for name, module in MODEL.named_modules():
        layer_names.append(name)

    return layer_names[2:]

def get_model(name):
    assert name == 'Yassine_test_1'
    # Create an instance of the custom model
    model = YassineTest1()

    # # Check if there is a .pth file in the directory and load the weights if found
    # weights_file = 'path_to_directory/my_custom_model_weights.pth'  # Replace with actual path
    # if os.path.isfile(weights_file):
    #     print(f"Loading weights from {weights_file}")
    #     state_dict = torch.load(weights_file)
    #     model.load_state_dict(state_dict)
    # else:
    #     print("No .pth file found, using default initialization.")
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='Yassine_test_1', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_bibtex(name):
    return """ xx """

if __name__ == "__main__":
    check_models.check_base_models(__name__)

# if __name__ == "__main__":
#     # Create an instance of the model
#     model = YassineTest1()

#     # Print the initial weights and biases of the layers
#     print("Initial weights and biases of linear1:")
#     print("Weights:\n", model.linear1.weight)
#     print("Biases:\n", model.linear1.bias)

#     print("\nInitial weights and biases of linear2:")
#     print("Weights:\n", model.linear2.weight)
#     print("Biases:\n", model.linear2.bias)


