from brainscore_vision.model_helpers.check_submission import check_models
from transformers import AutoImageProcessor, AutoModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
import functools
import torch
import torch.nn as nn


class DinoV2Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.hidden_states = {}

        for i in range(12):
            self.model.encoder.layer[i].register_forward_hook(self.get_hook(f'layer_{i}'))

    def get_hook(self, name):
        def hook(module, input, output):
            # output shape: (batch, tokens, dim)
            # take only CLS token at index 0
            self.hidden_states[name] = output[:, 0, :]  # CLS token
        return hook

    def forward(self, x):
        self.hidden_states = {}
        _ = self.model(x)
        return self.hidden_states

def get_model(name):
    assert name == "dinov2_base"
    model = DinoV2Wrapper()
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='dinov2_base', model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == "dinov2_base"
    layer_names = [f'encoder.layer.{i}' for i in range(12)] 
    return layer_names


def get_bibtex(model_identifier):
    return """
    @misc{oquab2023dinov2,
      title={DINOv2: Learning Robust Visual Features without Supervision},
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2023},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
    """


if __name__ == '__main__':
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    for name, layer in model.named_modules():
        print(name)
    check_models.check_base_models(__name__)


# def get_model(name):
#     assert name == "dinov2_base"
#     model = AutoModel.from_pretrained('facebook/dinov2-base')
#     preprocessing = functools.partial(load_preprocess_images, image_size=224)
#     wrapper = PytorchWrapper(identifier='dinov2_base', model=model, preprocessing=preprocessing, batch_size=4)
#     wrapper.image_size = 224
#     return wrapper

# def get_layers(name):
#     assert name == "dinov2_base"
#     layer_names = []
#     for i in range(12):
#         layer_names.append(f"encoder.layer.{i}.attention.output.dense")  # Attention output
#     layer_names.append("layernorm")  # Final output representation
#     return layer_names


# def get_bibitex(model_identifier):
#     return """
#     misc{oquab2023dinov2,
#       title={DINOv2: Learning Robust Visual Features without Supervision}, 
#       author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
#       year={2023},
#       eprint={2304.07193},
#       archivePrefix={arXiv},
#       primaryClass={cs.CV}
# }
# """

# if __name__ == '__main__':
#     check_models.check_base_models(__name__)
