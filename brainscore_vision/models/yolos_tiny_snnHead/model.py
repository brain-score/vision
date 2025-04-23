from brainscore_vision.model_helpers.check_submission import check_models
import functools
import numpy as np
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image
import snntorch as snn
import torch.nn as nn
from snntorch import surrogate

class PytorchWrapperFixed(PytorchWrapper):
    @staticmethod
    def _tensor_to_numpy(output):
        if isinstance(output, tuple):
            output = output[0]
        return output.cpu().data.numpy()

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if isinstance(output, tuple):
                output = output[0]
            target_dict[name] = PytorchWrapperFixed._tensor_to_numpy(output)
        hook = layer.register_forward_hook(hook_function)
        return hook

    # def look_at(self, stimuli, layers):
    #     activations = super().look_at(stimuli, layers)
    #     print("\n--- Activation Coordinate Debug ---")
    #     for layer, act in activations.items():
    #         print(f"Layer: {layer}")
    #         print(f"Coords: {list(act.coords.keys())}")
    #         print(f"Shape: {act.shape}")
    #         print(f"Coords Detail: {act.coords}\n")

    #     # Fix missing 'embedding' coordinate in each DataArray
    #     for layer, act in activations.items():
    #         if 'embedding' not in act.coords and 'neuroid' in act.coords:
    #             act.coords['embedding'] = ('neuroid', act.coords['neuroid'].values)
    #     return activations


class YolosSpikingHead(nn.Module):
    def __init__(self, input_dim, num_classes, beta=0.5, spike_grad=surrogate.fast_sigmoid(slope=25)):
        super().__init__()
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(input_dim, 256)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(256, num_classes + 4)  # classes + bounding box coordinates
        
        # Initialize membrane potential
        self.mem1 = self.mem2 = None
        
    def forward(self, x, num_steps=5):
        # Reset membrane potential at the start of each forward pass
        self.mem1 = self.mem2 = None
        
        outputs = []
        # Run for multiple time steps
        for _ in range(num_steps):
            # First spiking layer
            cur1 = self.fc1(x)
            spk1, self.mem1 = self.lif1(cur1, self.mem1)
            
            # Second spiking layer
            cur2 = self.fc2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)
            
            outputs.append(spk2)
        
        # Average across time steps
        return torch.stack(outputs).mean(0)

class YolosWithSpikingHead(nn.Module):
    def __init__(self, yolos_model, num_classes=91):
        super().__init__()
        self.backbone = yolos_model.vit
        
        # Get the correct input dimension by checking the model's configuration
        input_dim = yolos_model.config.hidden_size  # This will automatically get the correct dimension
        print(f"Using input dimension: {input_dim}")
        
        self.spiking_head = YolosSpikingHead(input_dim, num_classes)
        
    def forward(self, pixel_values):
        # Get features from YOLOS backbone
        features = self.backbone(pixel_values).last_hidden_state
        
        # Class token output (first token)
        cls_token = features[:, 0, :]
        
        # Pass through spiking head
        outputs = self.spiking_head(cls_token)
        
        # Split outputs into class logits and box coordinates
        class_logits, bbox_pred = outputs.split([91, 4], dim=1)
        
        return {'logits': class_logits, 'pred_boxes': bbox_pred}

def get_model_list():
    return ['yolos_tiny_snnHead']

def get_model(name):
    assert name == 'yolos_tiny_snnHead'
    image_size = 224  
    processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')
    yolos_model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    
    # Create the combined model with spiking head
    model = YolosWithSpikingHead(yolos_model)
    print_layer_names(model)
    
    preprocessing = functools.partial(load_preprocess_images, processor=processor, image_size=image_size)
    wrapper = PytorchWrapperFixed(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper

def get_layers(name):
    assert name == "yolos_tiny_snnHead"
    layer_names = [
        # Original YOLOS layers
        "backbone.encoder.layer.0.attention.output.dense",
        "backbone.encoder.layer.0.output.dense",
        "backbone.encoder.layer.2.attention.output.dense",
        "backbone.encoder.layer.2.output.dense",
        "backbone.encoder.layer.4.attention.output.dense",
        "backbone.encoder.layer.4.output.dense",
        "backbone.encoder.layer.6.attention.output.dense",
        "backbone.encoder.layer.6.output.dense",
        "backbone.encoder.layer.8.attention.output.dense",
        "backbone.encoder.layer.8.output.dense",
        "backbone.encoder.layer.10.attention.output.dense",
        "backbone.encoder.layer.10.output.dense",
        "backbone.encoder.layer.11.attention.output.dense",
        "backbone.encoder.layer.11.output.dense",
        "backbone.layernorm",
        
        "spiking_head.lif1",  # The first spiking neuron layer
        "spiking_head.fc1",   # The linear transform before spiking
        "spiking_head.lif2",  # The second spiking neuron layer
        "spiking_head.fc2"
    ]
    return layer_names

def get_bibtex(model_identifier):
    return """@article{fang2021yolos,
      title={You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection},
      author={Tianyang Fang and Wen Wang and Bohan Zhuang and Chunhua Shen},
      journal={arXiv preprint arXiv:2106.00666},
      year={2021}
    }
    
    @article{eshraghian2023training,
      title={Training Spiking Neural Networks Using Lessons From Deep Learning},
      author={Eshraghian, Jason K. and Ward, Peggy and Neftci, Emre and Wang, Xinxin and Lenz, Gregor and Dwivedi, Girish and Bennamoun, Mohammed and Jeong, Doo Seok and Lu, Wei D.},
      journal={Proceedings of the IEEE},
      year={2023},
      volume={111},
      number={9},
      pages={936-979}
    }"""

def load_preprocess_images(image_filepaths, image_size, processor=None, **kwargs):
    images = [Image.open(filepath).convert('RGB') for filepath in image_filepaths]
    images = [image.resize((image_size, image_size)) for image in images]
    if processor is not None:
        images = [processor(images=image, return_tensors="pt", **kwargs)['pixel_values'] for image in images]
        images = torch.cat(images).cpu().numpy()
    return images

# Add a function to extract spiking embeddings
def get_spiking_embeddings(model, images):
    """Extract spiking embeddings from the model's spiking head"""
    with torch.no_grad():
        # Process images
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        # Forward pass through backbone
        features = model.backbone(images).last_hidden_state
        
        # Get class token
        cls_token = features[:, 0, :]
        
        # Run through first spiking layer only
        model.spiking_head.mem1 = None
        cur1 = model.spiking_head.fc1(cls_token)
        spk1, _ = model.spiking_head.lif1(cur1, model.spiking_head.mem1)
        # The spiking activity is the embedding
        return spk1.cpu().numpy()
    
def print_layer_names(model):
    """Utility function to print all layer names and their types in a model"""
    print("Model layers:")
    for name, module in model.named_modules():
        print(f"Layer name: {name}, Type: {type(module).__name__}")

    
if __name__ == '__main__':
    check_models.check_base_models(__name__)
