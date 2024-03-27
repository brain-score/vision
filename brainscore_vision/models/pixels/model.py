from collections import OrderedDict

import numpy as np
from PIL import Image

from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper
from brainscore_vision.model_helpers.check_submission import check_models


class PixelModel:
    def __init__(self):
        self._extractor = ActivationsExtractorHelper(identifier='pixels', preprocessing=None,
                                                     get_activations=self._pixels_from_paths)
        self._extractor.insert_attrs(self)

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    # Note that we cannot assign __call__ as attribute due to Python convention, 
    # so we cannot set `self.__call__ = self._extractor` in the `__init__` and 
    # instead have to explicitly implement the function here.
    def __call__(self, *args, **kwargs):
        return self._extractor(*args, **kwargs)

    def _pixels_from_paths(self, paths, layer_names):
        np.testing.assert_array_equal(layer_names, ['pixels'])
        pixels = [self._parse_image(path) for path in paths]
        return OrderedDict([('pixels', np.array(pixels))])

    def _parse_image(self, path):
        image = Image.open(path)
        # make sure everything is in RGB and not grayscale L
        image = image.convert('RGB')
        image = image.resize((256, 256))  # resize all images to same size
        return np.array(image)


# Main Method: In submitting a custom model, you should not have to mess
# with this.
if __name__ == "__main__":
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
