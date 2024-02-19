from collections import OrderedDict

import numpy as np

from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper


class KerasWrapper:
    def __init__(self, model, preprocessing, identifier=None, *args, **kwargs):
        """
        :param model: a keras model with a function `preprocess_input`
            that will later be called on the loaded numpy image
        """
        self._model = model
        identifier = identifier or model.name
        self._extractor = ActivationsExtractorHelper(
            identifier=identifier, get_activations=self.get_activations, preprocessing=preprocessing,
            *args, **kwargs)
        self._extractor.insert_attrs(self)

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    def set_visual_degrees(self, visual_degrees):
        """A method to allow the ModelCommitment to set the visual angle of the ActivationsExtractorHelper."""
        self._extractor.set_visual_degrees(visual_degrees)

    def get_activations(self, images, layer_names):
        from keras import backend as K
        input_tensor = self._model.input
        layers = [layer for layer in self._model.layers if layer.name in layer_names]
        layers = sorted(layers, key=lambda layer: layer_names.index(layer.name))
        if 'logits' in layer_names:
            layers.insert(layer_names.index('logits'), self._model.layers[-1])
        assert len(layers) == len(layer_names)
        layer_out_tensors = [layer.output for layer in layers]
        functor = K.function([input_tensor] + [K.learning_phase()], layer_out_tensors)  # evaluate all tensors at once
        layer_outputs = functor([images, 0.])  # 0 to signal testing phase
        return OrderedDict([(layer_name, layer_output) for layer_name, layer_output in zip(layer_names, layer_outputs)])

    def __repr__(self):
        return repr(self._model)

    def graph(self):
        import networkx as nx
        g = nx.DiGraph()
        for layer in self._model.layers:
            g.add_node(layer.name, object=layer, type=type(layer))
            for outbound_node in layer._outbound_nodes:
                g.add_edge(layer.name, outbound_node.outbound_layer.name)
        return g


def load_images(image_filepaths, image_size):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = [scale_image(image, image_size) for image in images]
    return np.array(images)


def load_image(image_filepath):
    try:  # keras API before tensorflow 2.9.1
        from keras.preprocessing.image import load_img
        from keras.preprocessing.image import img_to_array
    except ImportError:
        from tensorflow.keras.utils import load_img
        from tensorflow.keras.utils import img_to_array
    img = load_img(image_filepath)
    x = img_to_array(img)
    return x


def scale_image(img, image_size):
    from PIL import Image
    try:  # keras API before tensorflow 2.9.1
        from keras.preprocessing.image import img_to_array
    except ImportError:
        from tensorflow.keras.utils import img_to_array
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((image_size, image_size))
    img = img_to_array(img)
    return img


def preprocess(image_filepaths, image_size, *args, **kwargs):
    # only a wrapper to avoid top-level keras imports
    from keras.applications.imagenet_utils import preprocess_input
    images = load_images(image_filepaths, image_size=image_size)
    return preprocess_input(images, *args, **kwargs)
