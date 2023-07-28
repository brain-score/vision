import functools

# from brainscore import score_model
# from brainscore.benchmarks import public_benchmark_pool
import torchvision.models
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images

# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models
import open_clip

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# def score(model):
#     # score = score_model(model_identifier='open_clip', model=model, benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
#     benchmark = public_benchmark_pool['movshon.FreemanZiemba2013public.aperture.IT-pls']
#     # model = my_model()
#     score = benchmark(model)
#     print('Score: ', score)

def get_model_list():
    return ['open_clip']

def get_model(name):
    assert name == 'open_clip'
    model, preprocess = open_clip.create_model_from_pretrained('ViT-B-32', pretrained='laion2b_e16')
    preprocessing = functools.partial(load_preprocess_images, image_size=224)

    wrapper = PytorchWrapper(identifier='open_clip', model=model.visual, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == 'open_clip'
    return ['conv1']


def get_bibtex(model_identifier):
    return ''

if __name__ == '__main__':
    check_models.check_base_models(__name__)
    # model = get_model('open_clip')
    # score(model)
    # wrapper = get_model('open_clip')
    # layers = wrapper.layers()
    # for i in range(198):
    #     print(i)
    #     print(next(layers))