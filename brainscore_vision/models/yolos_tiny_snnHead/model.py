from brainscore_vision.model_helpers.check_submission import check_models
import functools
import numpy as np
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image
import snntorch as snn
import torch.nn as nn

# class PytorchWrapperFixed(PytorchWrapper):
#     @staticmethod
#     def _tensor_to_numpy(output):
#         if isinstance(output, tuple):
#             output = output[0]
#         return output.cpu().data.numpy()

#     def register_hook(self, layer, layer_name, target_dict):
#         def hook_function(_layer, _input, output, name=layer_name):
#             if isinstance(output, tuple):
#                 output = output[0]
#             target_dict[name] = PytorchWrapperFixed._tensor_to_numpy(output)
#         hook = layer.register_forward_hook(hook_function)
#         return hook
class PytorchWrapperFixed(PytorchWrapper):
    def __init__(self, *args, spiking_head=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.spiking_head = spiking_head

    @staticmethod
    def _tensor_to_numpy(output):
        if isinstance(output, tuple):
            output = output[0]
        return output.cpu().data.numpy()

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if isinstance(output, tuple):
                output = output[0]
            if name == "vit.layernorm" and self.spiking_head is not None:
                # output shape: [batch_size, seq_len, hidden_dim]
                # e.g., [1, 197, 768] — let's use the [CLS] token (index 0)
                cls_token = output[:, 0, :]  # shape: [batch_size, hidden_dim]
                # print("CLS shape:", cls_token.shape)
                spiking_out = self.spiking_head(cls_token)
                output = spiking_out
            target_dict[name] = PytorchWrapperFixed._tensor_to_numpy(output)
        hook = layer.register_forward_hook(hook_function)
        return hook

class SpikingHead(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=256, output_dim=128, beta=0.95, num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta)
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_out = []
        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_out.append(spk2)

        # Collapse over time steps (e.g., mean firing rate)
        return torch.stack(spk_out).mean(dim=0)

def get_model_list():
    return ['yolos_tiny_snnHead']

def get_model(name):
    assert name == 'yolos_tiny_snnHead'
    image_size = 224
    processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    hidden_dim = model.config.hidden_size  # This will be 192 for YOLOS-Tiny
    spiking_head = SpikingHead(input_dim=hidden_dim)
 

    preprocessing = functools.partial(load_preprocess_images, processor=processor, image_size=image_size)
    wrapper = PytorchWrapperFixed(identifier=name, model=model, preprocessing=preprocessing, spiking_head=spiking_head)
    wrapper.image_size = image_size
    return wrapper

def get_layers(name):
    assert name == "yolos_tiny_snnHead"
    layer_names = [
        "vit.encoder.layer.0.attention.output.dense",
        "vit.encoder.layer.0.output.dense",
        "vit.encoder.layer.2.attention.output.dense",
        "vit.encoder.layer.2.output.dense",
        "vit.encoder.layer.4.attention.output.dense",
        "vit.encoder.layer.4.output.dense",
        "vit.encoder.layer.6.attention.output.dense",
        "vit.encoder.layer.6.output.dense",
        "vit.encoder.layer.8.attention.output.dense",
        "vit.encoder.layer.8.output.dense",
        "vit.encoder.layer.10.attention.output.dense",
        "vit.encoder.layer.10.output.dense",
        "vit.encoder.layer.11.attention.output.dense",
        "vit.encoder.layer.11.output.dense",

        "vit.layernorm",

        "class_labels_classifier.layers.0",
        "class_labels_classifier.layers.1",
        "class_labels_classifier.layers.2",
    ]
    return layer_names

def get_bibtex(model_identifier):
    return """@article{fang2021yolos,
      title={You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection},
      author={Tianyang Fang and Wen Wang and Bohan Zhuang and Chunhua Shen},
      journal={arXiv preprint arXiv:2106.00666},
      year={2021}
    }"""

def load_preprocess_images(image_filepaths, image_size, processor=None, **kwargs):
    images = [Image.open(filepath).convert('RGB') for filepath in image_filepaths]
    images = [image.resize((image_size, image_size)) for image in images]
    if processor is not None:
        images = [processor(images=image, return_tensors="pt", **kwargs)['pixel_values'] for image in images]
        images = torch.cat(images).cpu().numpy()
    return images

if __name__ == '__main__':
    check_models.check_base_models(__name__)
