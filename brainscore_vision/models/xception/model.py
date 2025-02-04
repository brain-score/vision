import functools
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import timm


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'xception'
    model = timm.create_model('xception', pretrained=True)
    preprocessing = functools.partial(load_preprocess_images, image_size=299)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing, batch_size=4)
    wrapper.image_size = 299
    return wrapper


def get_layers(name):
    assert name == 'xception'
    layer_names = (
        # Block 3 (2 layers)
        [f'block3.rep.{i}.pointwise' for i in [1, 4]] +
        # Block 4 (2 layers)
        [f'block4.rep.{i}.pointwise' for i in [1, 4]] +
        # Blocks 5-11 (3 layers each)
        [f'block{block}.rep.{layer}.pointwise'
         for block in range(5, 12)
         for layer in [1, 4, 7]] +
        # Block 12 (2 layers)
        [f'block12.rep.{i}.pointwise' for i in [1, 4]] +
        # Final layers
        ['conv3.pointwise', 'conv4.pointwise', 'global_pool.pool']
    )
    return layer_names


def get_bibtex(name):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return '''
@article{DBLP:journals/corr/ZagoruykoK16,
@misc{chollet2017xception,
      title={Xception: Deep Learning with Depthwise Separable Convolutions}, 
      author={François Chollet},
      year={2017},
      eprint={1610.02357},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
'''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
