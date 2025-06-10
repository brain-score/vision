import functools
import numpy as np
import torch
from transformers import AutoModelForImageSegmentation
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image
import torch.nn.functional as F
from brainscore_vision.model_helpers.check_submission import check_models

class briaa_rmbg_PytorchWrapper(PytorchWrapper):
    
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)  # retrieve original output
        if 'logits' in kwargs.get('layers', []):
            result = result.isel(neuroid=slice(1, None))  # remove background class in last layer
        return result
    @staticmethod
    def _tensor_to_numpy(output):
        if isinstance(output, tuple):
            output = output[0]
        return output.cpu().data.numpy()

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if isinstance(output, tuple):
                output = output[0]
            target_dict[name] = briaa_rmbg_PytorchWrapper._tensor_to_numpy(output)
        hook = layer.register_forward_hook(hook_function)
        return hook

def get_model_list():
    return ['briaai_rmbg_1_4']

def get_model(name):
    assert name == 'briaai_rmbg_1_4'
    image_size = 224  # Set the input size for the model
    model = AutoModelForImageSegmentation.from_pretrained('briaai/rmbg-1.4', trust_remote_code=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = briaa_rmbg_PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper

def get_layers(name):
    assert name == "briaai_rmbg_1_4"
    layer_names = []
    for i in range(1,6):
        layer_names.append(f"stage{i}.rebnconvin.conv_s1")
        layer_names.append(f"stage{i}.rebnconv1.conv_s1")
        layer_names.append(f"stage{i}.rebnconvin.bn_s1")
        layer_names.append(f"stage{i}.rebnconv1.bn_s1")
        layer_names.append(f"stage{i}.rebnconvin.relu_s1")
        layer_names.append(f"stage{i}.rebnconv1.relu_s1")
    for i in range(1,3):
        layer_names.append(f"stage{i}.pool1")
        layer_names.append(f"stage{i}.pool2")
    layer_names.append("stage1.pool3")
    layer_names.append("stage1.pool4")
    layer_names.append("stage1.pool5")
    layer_names.append("stage2.pool3")
    layer_names.append("stage2.pool4")
    layer_names.append("stage3.pool3")
    return layer_names

def get_bibtex(model_identifier):
    return """"""

def load_preprocess_images(image_filepaths, image_size, **kwargs):
    images = [Image.open(filepath).convert('RGB') for filepath in image_filepaths]
    images = [image.resize((image_size, image_size)) for image in images]
    
    # Manually preprocess the images (without AutoProcessor)
    processed_images = []
    for image in images:
        image = np.array(image)  # Convert to numpy array (H, W, C)
        image = preprocess_image(image, model_input_size=(image_size, image_size))  # Preprocess the image
        processed_images.append(image)

    processed_images = [image.squeeze(0) for image in processed_images]
    
    return torch.stack(processed_images)

# Helper function for manual preprocessing (resize, normalize)
def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]  # Add channel dimension if missing
    
    # Convert to CHW format (channels, height, width)
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    
    # Add a batch dimension (batch_size, channels, height, width)
    im_tensor = im_tensor.unsqueeze(0)
    # Resize to the input size expected by the model
    im_tensor = F.interpolate(im_tensor, size=model_input_size, mode='bilinear', align_corners=False)
    
    # Normalize the image (assuming ImageNet-style normalization)
    image = torch.divide(im_tensor, 255.0)  # Normalize to [0, 1]
    image = (image - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return image

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)  # Resize back to original size
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)  # Normalize to [0, 1]
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)  # Convert to [0, 255]
    im_array = np.squeeze(im_array)  # Remove unnecessary dimensions
    return im_array

if __name__ == '__main__':
    check_models.check_base_models(__name__)
