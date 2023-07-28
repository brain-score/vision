# +
from model_tools.check_submission import check_models
from timm import create_model
import functools
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.check_submission import check_models
from model_tools.brain_transformation import ModelCommitment

MODEL_NAME = 'convnext_small'
LAYERS = ['stages.3.blocks.2.mlp.fc2','stages.3.blocks.2.mlp.fc1','stages.3.blocks.1.mlp.fc1','stages.3.blocks.1.mlp.fc2','stages.2.blocks.2.mlp.fc1','stages.2.blocks.1.mlp.fc1','stages.2.blocks.0.mlp.fc1','stages.1.blocks.2.mlp.fc2']

# +
# Create model
model = create_model(MODEL_NAME, pretrained=True)
# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_images, image_size=224)

# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier='convnext_small', model=model, preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='convnext_small', activations_model=activations_model,
                        # specify layers to consider
                        layers=LAYERS)


# -

def get_model_list():
    return ['convnext_small']


def get_model(name):
    assert name == 'convnext_small'    
    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'convnext_small'
    return LAYERS


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
