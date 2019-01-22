from collections import OrderedDict


class TensorflowSlimWrapper:
    def __init__(self, model):
        self._model = model

    def _load_images(self, image_filepaths, image_size):
        return image_filepaths

    def _load_image(self, image_filepath):
        import tensorflow as tf
        image = tf.read_file(image_filepath)
        image = tf.image.decode_png(image, channels=3)
        return image

    def _preprocess_images(self, images, image_size):
        return images

    def _preprocess_image(self, image, image_size):
        import tensorflow as tf
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, (image_size, image_size))
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        return image

    def __call__(self, images, layer_names):
        inputs, endpoints, sess = self._model.inputs, self._model.endpoints, self._model.session
        layer_tensors = OrderedDict((layer, endpoints[layer]) for layer in layer_names)
        layer_outputs = sess.run(layer_tensors, feed_dict={inputs: images})
        return layer_outputs

    def graph(self):
        import networkx as nx
        g = nx.DiGraph()
        for name, layer in self.endpoints.items():
            g.add_node(name, object=layer, type=type(layer))
        g.add_node("logits", object=self._logits, type=type(self._logits))
        return g


class TensorflowSlimContainer:
    def __init__(self, model_ctr, batch_size, image_size):
        import tensorflow as tf
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3])
        self.logits, self.endpoints = model_ctr(self.inputs)
