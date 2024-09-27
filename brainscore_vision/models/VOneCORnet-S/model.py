import functools
from vonenet import get_model as create_model
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images, PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models

def vonecornet(model_name='cornets'):
    model = create_model(model_name)
    model = model.module
    
    preprocessing = functools.partial(load_preprocess_images, image_size=224,
                                      normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    wrapper = PytorchWrapper(identifier='vone'+model_name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper


def get_model(name):
    assert name == "VOneCORnet-S"
    return vonecornet('cornets')

def get_layers(name):
    assert name == "VOneCORnet-S"
    model = create_model('cornets')
    model = model.module
    return list(dict(model.named_modules()).keys())[1:]


def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''' '''

if __name__ == '__main__':
    check_models.check_base_models(__name__)
