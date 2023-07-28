from model_tools.activations.tensorflow import load_resize_image
import tensorflow as tf
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.tensorflow import TensorflowSlimWrapper
from model_tools.check_submission import check_models
import numpy as np
import functools
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images


slim = tf.contrib.slim
tf.reset_default_graph()

image_size = 224
placeholder = tf.placeholder(dtype=tf.string, shape=[64])
preprocess = lambda image_path: load_resize_image(image_path, image_size)
preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)


with tf.variable_scope('my_model', values=[preprocess]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
        net = slim.conv2d(preprocess, 64, [11, 11], 4, padding='VALID', scope='conv1')
        net = slim.max_pool2d(net, [5, 5], 5, scope='pool1')
        net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
        net = slim.flatten(net, scope='flatten')
        net = slim.fully_connected(net, 1000, scope='logits')
        endpoints = slim.utils.convert_collection_to_dict(end_points_collection)

session = tf.Session()
session.run(tf.initialize_all_variables())



activations_model_tf = TensorflowSlimWrapper(identifier='2023_2_9_tf_2', labels_offset=0,
                                             endpoints=endpoints, inputs=placeholder, session=session)



model = ModelCommitment(identifier='2023_2_9_tf_2', activations_model=activations_model_tf,
                        # specify layers to consider
                        layers=['my_model/conv1', 'my_model/pool1', 'my_model/pool2'])


def get_model_list():
    print('This is a model for 2023_2_9_tf_2')
    return ['2023_2_9_tf_2']


def get_model(name):
    assert name == '2023_2_9_tf_2'
    wrapper = activations_model_tf
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == '2023_2_9_tf_2'
    return ['my_model/conv1', 'my_model/pool1', 'my_model/pool2']


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

if __name__ == '__main__':
    check_models.check_base_models(__name__)
