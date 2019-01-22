import logging
import numpy as np
from collections import OrderedDict

_logger = logging.getLogger(__name__)


class KerasWrapper:
    def __init__(self, model):
        """
        :param model: a keras model with a function `preprocess_input`
            that will later be called on the loaded numpy image
        """
        self._model = model

    def _load_image(self, image_filepath):
        from keras.preprocessing import image
        img = image.load_img(image_filepath)
        x = image.img_to_array(img)
        return x

    def _preprocess_images(self, images, image_size):
        images = [self._preprocess_image(image, image_size) for image in images]
        return np.array(images)

    def _preprocess_image(self, img, image_size):
        from PIL import Image
        from keras.preprocessing import image
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((image_size, image_size))
        img = image.img_to_array(img)
        img = self._model.preprocess_input(img)
        return img

    def __call__(self, images, layer_names):
        from keras import backend as K
        input_tensor = self._model.input
        layers = [layer for layer in self._model.layers if layer.name in layer_names]
        layers = sorted(layers, key=lambda layer: layer_names.index(layer.name))
        layer_out_tensors = [layer.output for layer in layers]
        functor = K.function([input_tensor] + [K.learning_phase()], layer_out_tensors)  # evaluate all tensors at once
        layer_outputs = functor([images, 0.])  # 0 to signal testing phase
        return OrderedDict([(layer_name, layer_output) for layer_name, layer_output
                            in zip([layer.name for layer in layers], layer_outputs)])

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


def preprocess(*args, **kwargs):
    # only a wrapper to avoid top-level keras imports
    from keras.applications.imagenet_utils import preprocess_input
    return preprocess_input(*args, **kwargs)
