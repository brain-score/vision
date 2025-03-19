from brainscore_vision.model_helpers.check_submission import check_models
import functools
from transformers import AutoImageProcessor, AutoModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

def get_model(name):
    assert name == "dinov2_base"
    
    model = AutoModel.from_pretrained('facebook/dinov2-base')
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='dinov2_base', model=model, preprocessing=preprocessing, batch_size=4)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == "dinov2_base"
    layer_names = [
        "embeddings.patch_embeddings.projection",  # Early feature extraction
    ]
    # Add layers for each transformer block (0-11)
    for i in range(12):
        layer_names.append(f"encoder.layer.{i}.attention.output.dense")  # Attention output
        layer_names.append(f"encoder.layer.{i}.mlp.fc2")                 # MLP output
    
    layer_names.append("layernorm")  # Final output representation

    # for n,_ in AutoModel.from_pretrained('facebook/dinov2-base').named_modules():
    #     print(n)
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
    # get_layers("dinov2-base")
    check_models.check_base_models(__name__)
