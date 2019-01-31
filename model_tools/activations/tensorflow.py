from collections import OrderedDict

from model_tools.activations.core import ActivationsExtractorHelper


class TensorflowWrapper:
    def __init__(self, identifier, inputs, endpoints: dict, session, logits=None, *args, **kwargs):
        import tensorflow as tf
        self._inputs = inputs
        self._endpoints = endpoints
        self._session = session or tf.Session()
        self._extractor = ActivationsExtractorHelper(identifier=identifier, get_activations=self.get_activations,
                                                     preprocessing=None, *args, **kwargs)
        self._extractor.insert_attrs(self)

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    def get_activations(self, images, layer_names):
        layer_tensors = OrderedDict((layer, self._endpoints[
            layer if (layer != 'logits' or layer in self._endpoints) else next(reversed(self._endpoints))])
                                    for layer in layer_names)
        layer_outputs = self._session.run(layer_tensors, feed_dict={self._inputs: images})
        return layer_outputs

    def graph(self):
        import networkx as nx
        g = nx.DiGraph()
        for name, layer in self._endpoints.items():
            g.add_node(name, object=layer, type=type(layer))
        g.add_node("logits", object=self.logits, type=type(self.logits))
        return g


class TensorflowSlimWrapper(TensorflowWrapper):
    def __init__(self, *args, labels_offset=1, **kwargs):
        super(TensorflowSlimWrapper, self).__init__(*args, **kwargs)
        self._labels_offset = labels_offset

    def get_activations(self, images, layer_names):
        layer_outputs = super(TensorflowSlimWrapper, self).get_activations(images, layer_names)
        if 'logits' in layer_outputs:
            layer_outputs['logits'] = layer_outputs['logits'][:, self._labels_offset:]
        return layer_outputs


def load_image(image_filepath):
    import tensorflow as tf
    image = tf.read_file(image_filepath)
    image = tf.image.decode_png(image, channels=3)
    return image


def resize_image(image, image_size):
    import tensorflow as tf
    image = tf.image.resize_images(image, (image_size, image_size))
    return image


def load_resize_image(image_path, image_size):
    image = load_image(image_path)
    image = resize_image(image, image_size)
    return image
