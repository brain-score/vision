import keras.applications
import tensorflow as tf
from model_tools.activations.keras import load_images, KerasWrapper
from model_tools.check_submission import check_models
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, activations

from brainscore_vision.model_helpers import download_weights


#   Based on CORnet-S.
#   Source: https://github.com/dicarlolab/CORnet
class RecurrentCNNBlock(layers.Layer):

    scale = 2

    def __init__(self, out_channels, times=1, use_fb=False, use_fb2=False, fb_channels=None, us_factor=None, fb2_channels=None, us2_factor=None):
        super(RecurrentCNNBlock, self).__init__()

        self.hidden_state = None

        self.times = times
        self.use_fb = use_fb
        self.conv_input = layers.Conv2D(out_channels, kernel_size=1, padding="same", use_bias=False)
        self.skip = layers.Conv2D(out_channels, kernel_size=1, strides=(1,1), padding="same", use_bias=False)
        self.norm_skip = tf.contrib.layers.instance_norm()

        self.conv1 = layers.Conv2D(out_channels * self.scale, kernel_size=1, padding="same", use_bias=False)
        self.nonlin1 = layers.Activation(activations.relu)
        self.conv2 = layers.Conv2D(out_channels * self.scale, kernel_size=3, strides=(1,1), padding="same", use_bias=False)
        self.nonlin2 = layers.Activation(activations.relu)
        self.conv3 = layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False)
        self.nonlin3 = layers.Activation(activations.relu)

        if use_fb:
            # self.upsample = layers.UpSampling2D(size=(us_factor, us_factor))
            self.conv_fb = layers.Conv2DTranspose(fb_channels, kernel_size=3, strides=(us_factor,us_factor), padding="same", use_bias=False)

        self.norm_fb = tf.contrib.layers.instance_norm()
        self.conv_lat = layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False)

        # if use_fb2:
        #     self.conv_fb2 = layers.Conv2DTranspose(fb2_channels, kernel_size=3, strides=(us2_factor,us2_factor), padding="same", use_bias=False)
        #     self.norm_fb2 = tf.contrib.layers.instance_norm()

        for t in range(self.times):
            setattr(self, f'norm1_{t}', tf.contrib.layers.instance_norm())
            setattr(self, f'norm2_{t}', tf.contrib.layers.instance_norm())
            setattr(self, f'norm3_{t}', tf.contrib.layers.instance_norm())


    #   Recurrent CNN call.
    #   Sourced:  https://github.com/cjspoerer/rcnn-sat
    def call(self, inp, td_inp=None, fb=False, fb2=False, training=False):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.strides = (1,1)
            else:
                skip = x
                self.conv2.strides = (1,1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)
            x += skip

            # if t != self.times-1:
            x = self.nonlin3(x)

        #   First step of recurrence so no top-down or lateral input yet.
        if self.hidden_state is None:
            # x = self.nonlin3(x)
            self.hidden_state = x
        else:
            x += self.conv_lat(self.hidden_state)
            if td_inp is not None:
                x += self.conv_fb(td_inp)

            x = self.norm_fb(x)
            x = self.nonlin3(x)
            self.hidden_state = x

        return x


class RecurrentCNN(keras.Model):
    def __init__(self, fb_loops=0):
        super(RecurrentCNN, self).__init__()

        self.fb_loops = fb_loops

        self.V1 = [
            layers.Conv2D(64, kernel_size=7, strides=(1,1), padding="same", use_bias=False),
            tf.contrib.layers.instance_norm(),
            layers.Activation(activations.relu),
            layers.MaxPooling2D(pool_size=(2,2), padding="same"),
            layers.Conv2D(64, kernel_size=3, strides=(1,1), padding="same", use_bias=False),
            tf.contrib.layers.instance_norm(),
            layers.Activation(activations.relu)
        ]

        self.nonlin = layers.Activation(activations.relu)

        self.V1_conv_fb1 = layers.Conv2DTranspose(64, kernel_size=3, strides=(1,1), padding="same", use_bias=False)
        self.V1_conv_fb2 = layers.Conv2DTranspose(64, kernel_size=3, strides=(4,4), padding="same", use_bias=False)
        self.V1_norm_fb = tf.contrib.layers.instance_norm()
        self.V1_conv_lat = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False)

        self.V2 = RecurrentCNNBlock(128, times=1, use_fb=True, fb_channels=128, us_factor=2)
        self.mp1 = layers.MaxPooling2D(pool_size=(2,2), padding="same")
        self.V4 = RecurrentCNNBlock(256, times=1, use_fb=True, fb_channels=256, us_factor=2)
        self.mp2 = layers.MaxPooling2D(pool_size=(2,2), padding="same")
        self.IT = RecurrentCNNBlock(512, times=1, use_fb=False)

        self.decoder = [
            layers.GlobalAveragePooling2D(),
            layers.Dense(9, activation='softmax')
        ]


    #   Recurrent CNN call.
    def call(self, x, training=None, mask=None):

        for stage in self.V1:
            x = stage(x)

        #   First pass to save initial states.
        v1_response = x
        v1_state = x
        x = self.V2(x)
        x = self.mp1(x)
        x = self.V4(x)
        # v4_response = x
        # v4_state = x
        x = self.mp2(x)
        x = self.IT(x)


        for t in range(self.fb_loops):
            x = v1_response + self.V1_conv_lat(v1_state) + self.V1_conv_fb1(self.V2.hidden_state) + self.V1_conv_fb2(self.IT.hidden_state)
            x = self.V1_norm_fb(x)
            x = self.nonlin(x)
            v1_state = x

            x = self.V2(x, td_inp=self.V4.hidden_state)
            x = self.mp1(x)
            # x = v4_response + self.V4.conv_lat(v4_state) + self.V4.conv_fb(self.IT.hidden_state)
            # x = self.V4.norm_fb(x)
            # x = self.nonlin(x)
            # v4_state = x
            x = self.V4(x, td_inp=self.IT.hidden_state)
            x = self.mp2(x)
            x = self.IT(x)

        #   Reset all hidden states to None here.
        self.V2.hidden_state = None
        self.V4.hidden_state = None
        self.IT.hidden_state = None

        for stage in self.decoder:
            x = stage(x)

        return x


def get_model_list():
    return ['sketch_model-4o-ep10']


def get_model(name):
    assert name == 'sketch_model-4o-ep10'
    model = RecurrentCNN(fb_loops=4)
    download_weights(
        bucket='brainscore-vision', folder_path='models/4o_model_submission/rcnn_cand_4o_weights_10ep',
        filename_version_sha=[
            ('.data-00000-of-00001', 'yYBXN7uf57Y70EdQLchC_dMU5KO6GkIi', 'fef2a64f8c591f5d7562677272c91dcf88989d53'),
            ('.index', '7SJtRE0pdSahjajwjW5c_nVPVGBKn3q2', 'ba9e531243cb87de8562152ab8a22cdb3d218c3b'),
            ('checkpoint', '7rAEuKAwsWAmjT9OvXMsPOaQkogaxMMC', 'a39e7f800a473d18781931c26f84eb3399bdb484')],
        save_directory=Path(__file__).parent / 'models' / 'rcnn_cand_4o_weights_10ep')
    model.load_weights('rcnn_cand_4o_weights_10ep/')
    model_preprocessing = keras.applications.vgg16.preprocess_input
    load_preprocess = lambda image_filepaths: model_preprocessing(load_images(image_filepaths, image_size=32))
    wrapper = KerasWrapper(model, load_preprocess)
    wrapper.image_size = 32
    return wrapper


def get_layers(name):
    assert name == 'sketch_model-4o-ep10'
    return ["activation_2", "max_pooling2d_1", "max_pooling2d_2", "recurrent_cnn_block_2", "global_average_pooling2d"]


if __name__ == '__main__':
    check_models.check_base_models(__name__)

