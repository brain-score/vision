from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import ssl
from .helpers.hmax import HMAX
from .helpers.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file


ssl._create_default_https_context = ssl._create_unverified_context
model = None


def get_model(name):
    assert name == 'hmax'
    return get_hmax(name, 224)


def get_hmax(identifier, image_size):
    weights_path = load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",
                                    relative_path="hmax/universal_patch_set.mat",
                                    version_id="null",
                                    sha1="acc7316fcb0d1797486bb62753b71e158216a92a")
    global model 
    model = HMAX(str(weights_path))
    
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=identifier, model=model,
                             preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper


def get_layers(name):
    assert name == 'hmax'
    global model 
    layer_names = []
    for name, module in model.named_modules():
        print(name)
        layer_names.append(name)

    return layer_names[-8:]


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return """@ARTICLE {,
                author = {G. Cortelazzo and M. Balanza},
                journal = {IEEE Transactions on Pattern Analysis & Machine Intelligence},
                title = {Frequency Domain Analysis of Translations with Piecewise Cubic Trajectories},
                year = {1993},
                volume = {29},
                number = {04},
                issn = {1939-3539},
                pages = {411-416},
                keywords = {frequency domain motion analysis; motion estimation; translations; piecewise cubic trajectories; cubic spline trajectories; finite-duration effects; constant velocity motion; first-order model; frequency-domain analysis; motion estimation; splines (mathematics)},
                doi = {10.1109/34.206960},
                publisher = {IEEE Computer Society},
                address = {Los Alamitos, CA, USA},
                month = {apr}
                }
            """


if __name__ == '__main__':
    check_models.check_base_models(__name__)
