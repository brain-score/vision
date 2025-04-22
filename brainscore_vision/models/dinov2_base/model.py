from brainscore_vision.model_helpers.check_submission import check_models
from transformers import AutoImageProcessor, AutoModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

def get_model(name):
    assert name == "dinov2_base"

    model = AutoModel.from_pretrained('facebook/dinov2-base')
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')

    def preprocessing(images):
        # Assumes images are in PIL.Image format or a format processor accepts
        inputs = processor(images=images[0], return_tensors="pt")
        return inputs['pixel_values']  # what the model expects

    wrapper = PytorchWrapper(identifier='dinov2_base', model=model, preprocessing=preprocessing, batch_size=4)
    wrapper.image_size = 224
    return wrapper
    # assert name == "dinov2_base"
    # model = AutoModel.from_pretrained('facebook/dinov2-base')
    # preprocessing = functools.partial(load_preprocess_images, image_size=224)
    # wrapper = PytorchWrapper(identifier='dinov2_base', model=model, preprocessing=preprocessing, batch_size=4)
    # wrapper.image_size = 224
    # return wrapper

def get_layers(name):
    assert name == "dinov2_base"
    layer_names = []
    # Add layers for each transformer block (0-11)
    for i in range(12):
        layer_names.append(f"encoder.layer.{i}.attention.output.dense")  # Attention output
        layer_names.append(f"encoder.layer.{i}.mlp.fc2")                 # MLP output
    
    layer_names.append("layernorm")  # Final output representation

    return layer_names


def get_bibitex(model_identifier):
    return """
    misc{oquab2023dinov2,
      title={DINOv2: Learning Robust Visual Features without Supervision}, 
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2023},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

if __name__ == '__main__':
    # get_layers("dinov2_base")
    check_models.check_base_models(__name__)
    # Load the model
    # model = AutoModel.from_pretrained('facebook/dinov2-base')

    # # Create a dummy input (batch size 1, 3 channels, 224x224 image)
    # dummy_input = torch.randn(1, 3, 224, 224)

    # # Dictionary to store outputs
    # outputs = OrderedDict()

    # # Register hooks
    # def hook_fn(name):
    #     def hook(module, input, output):
    #         outputs[name] = output.shape if hasattr(output, 'shape') else str(type(output))
    #     return hook

    # # Attach hooks
    # for name, module in model.named_modules():
    #     module.register_forward_hook(hook_fn(name))

    # # Forward pass
    # with torch.no_grad():
    #     _ = model(dummy_input)

    # # Print output shapes
    # for name, shape in outputs.items():
    #     print(f"{name}: {shape}")
