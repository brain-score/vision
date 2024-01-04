from model_tools.activations.keras import load_images, KerasWrapper
import keras.applications
from model_tools.check_submission import check_models

# This is an example implementation for submitting vgg-16 as a keras model to brain-score
# If you use keras, don't forget to add it and its dependencies to the setup.py


def get_model_list():
    return ['vgg-16']


def get_model(name):
    assert name == 'vgg-16'
    model = keras.applications.vgg16.VGG16()
    model_preprocessing = keras.applications.vgg16.preprocess_input
    load_preprocess = lambda image_filepaths: model_preprocessing(load_images(image_filepaths, image_size=224))
    wrapper = KerasWrapper(model, load_preprocess)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'vgg-16'
    return [f'block{i + 1}_pool' for i in range(5)] + ['fc1', 'fc2']


def get_bibtex(model_identifier):
    assert model_identifier == 'vgg-16'
    return """@InProceedings{Simonyan15,
              author       = "Karen Simonyan and Andrew Zisserman",
              title        = "Very Deep Convolutional Networks for Large-Scale Image Recognition",
              booktitle    = "International Conference on Learning Representations",
              year         = "2015",
              url          = "https://arxiv.org/abs/1409.1556", 
            }"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)

