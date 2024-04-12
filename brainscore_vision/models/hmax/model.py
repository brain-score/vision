from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
import functools
from model_helpers.activations.pytorch import load_preprocess_images
import ssl
from .helpers.hmax import HMAX
from .helpers.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file


ssl._create_default_https_context = ssl._create_unverified_context


def get_model(name):
    return get_hmax(name, 224)


def get_hmax(identifier, image_size):
    weights_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                    relative_path="hmax/universal_patch_set.mat",
                                    version_id="fIX.lsvnc8qqjDr_sG_Dl9RyqWuG0OGC",
                                    sha1="acc7316fcb0d1797486bb62753b71e158216a92a")
    model = HMAX(str(weights_path))
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=identifier, model=model,
                             preprocessing=preprocessing, batch_size=2)
    wrapper.image_size = image_size
    return wrapper


def get_layers():
    return ['s1_out', 'c1_out', 'c2_out', 's2_out']


def get_bibtex():
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