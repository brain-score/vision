from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment


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
    return ["Yassine-test-1"]


MODEL = YassineTest1()

def get_layers(name):
    assert name == "Yassine-test-1"
    layer_names = []

    for name, module in MODEL.named_modules():
        layer_names.append(name)

    return layer_names[1:]

def get_model(name):
    assert name == 'Yassine-test-1'
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='Yassine-test-1', model=YassineTest1(), preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_bibtex(name):
    return """ xx """

if __name__ == "__main__":
    check_models.check_base_models(__name__)

# if __name__ == "__main__":
#     model = YassineTest1()
#     #print(model)
#     print(get_layers("Yassine-test-1"))
#     #print(get_model_list())
#     print("Model loaded successfully")


