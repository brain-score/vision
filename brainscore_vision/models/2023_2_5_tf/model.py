import logging
import os

#import s3

from model_tools.activations import TensorflowWrapper, TensorflowSlimWrapper
from model_tools.activations.keras import load_images, KerasWrapper
import keras.applications


# This is an example implementation for submitting resnet50 as a tensorflow SLIM model to brain-score
# If you use tensorflow, don't forget to add it and its dependencies to the setup.py
from model_tools.utils import fullname
from model_tools.check_submission import check_models


def get_model_list():
    print('This is a model for 2023_2_5_tf_1')
    return ['2023_2_5_tf_1']


def get_model(name):
    assert name == '2023_2_5_tf_1'
    model = TFSlimModel.init('2023_2_5_tf_1', net_name='resnet_v1_50', preprocessing_type='vgg',
                     image_size=224, labels_offset=0)
    model_preprocessing = keras.applications.resnet50.preprocess_input
    load_preprocess = lambda image_filepaths: model_preprocessing(load_images(image_filepaths, image_size=224))
    wrapper = KerasWrapper(model, load_preprocess)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == '2023_2_5_tf_1'
    return [f'block{i + 1}_pool' for i in range(5)] + ['fc1', 'fc2']


def get_bibtex(model_identifier):
    return """@article{DBLP:journals/corr/HeZRS15,
              author    = {Kaiming He and
                           Xiangyu Zhang and
                           Shaoqing Ren and
                           Jian Sun},
              title     = {Deep Residual Learning for Image Recognition},
              journal   = {CoRR},
              volume    = {abs/1512.03385},
              year      = {2015},
              url       = {http://arxiv.org/abs/1512.03385},
              archivePrefix = {arXiv},
              eprint    = {1512.03385},
              timestamp = {Wed, 17 Apr 2019 17:23:45 +0200},
              biburl    = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
              bibsource = {dblp computer science bibliography, https://dblp.org}
            }"""


class TFSlimModel:
    @staticmethod
    def init(identifier, preprocessing_type, image_size, net_name=None, labels_offset=1, batch_size=64,
             model_ctr_kwargs=None):
        import tensorflow as tf
        from nets import nets_factory

        tf.compat.v1.reset_default_graph()
        placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[batch_size])
        preprocess = TFSlimModel._init_preprocessing(placeholder, preprocessing_type, image_size=image_size)

        #net_name = net_name or identifier
        net_name = net_name
        model_ctr = nets_factory.get_network_fn(net_name, num_classes=labels_offset + 1000, is_training=False)
        logits, endpoints = model_ctr(preprocess, **(model_ctr_kwargs or {}))
        if 'Logits' in endpoints:  # unify capitalization
            endpoints['logits'] = endpoints['Logits']
            del endpoints['Logits']

        session = tf.compat.v1.Session()
        TFSlimModel._restore_imagenet_weights(identifier, session)
        wrapper = TensorflowSlimWrapper(identifier=identifier, endpoints=endpoints, inputs=placeholder, session=session,
                                        batch_size=batch_size, labels_offset=labels_offset)
        wrapper.image_size = image_size
        return wrapper

    @staticmethod
    def _init_preprocessing(placeholder, preprocessing_type, image_size):
        import tensorflow as tf
        from preprocessing import vgg_preprocessing, inception_preprocessing
        from model_tools.activations.tensorflow import load_image
        preprocessing_types = {
            'vgg': lambda image: vgg_preprocessing.preprocess_image(
                image, image_size, image_size, resize_side_min=image_size),
            'inception': lambda image: inception_preprocessing.preprocess_for_eval(
                image, image_size, image_size, central_fraction=None)
        }
        assert preprocessing_type in preprocessing_types
        preprocess_image = preprocessing_types[preprocessing_type]
        preprocess = lambda image_path: preprocess_image(load_image(image_path))
        preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)
        return preprocess

    @staticmethod
    def _restore_imagenet_weights(name, session):
        import tensorflow as tf
        var_list = None
        if name.startswith('mobilenet'):
            # Restore using exponential moving average since it produces (1.5-2%) higher accuracy according to
            # https://github.com/tensorflow/models/blob/a6494752575fad4d95e92698dbfb88eb086d8526/research/slim/nets/mobilenet/mobilenet_example.ipynb
            ema = tf.train.ExponentialMovingAverage(0.999)
            var_list = ema.variables_to_restore()
        restorer = tf.compat.v1.train.Saver(var_list)

        restore_path = './resnet_v1_50.ckpt' # TODO restore model weights
        restorer.restore(session, restore_path)

if __name__ == '__main__':
    check_models.check_base_models(__name__)
