from pathlib import Path

from model_tools.activations.keras import load_images, KerasWrapper
import keras.applications
from model_tools.check_submission import check_models

# This is an example implementation for submitting vgg-16 as a keras model to brain-score
# If you use keras, don't forget to add it and its dependencies to the setup.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
import os

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
        self.norm_skip = layers.BatchNormalization()

        self.conv1 = layers.Conv2D(out_channels * self.scale, kernel_size=1, padding="same", use_bias=False)
        self.nonlin1 = layers.Activation(activations.relu)
        self.conv2 = layers.Conv2D(out_channels * self.scale, kernel_size=3, strides=(1,1), padding="same", use_bias=False)
        self.nonlin2 = layers.Activation(activations.relu)
        self.conv3 = layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False)
        self.nonlin3 = layers.Activation(activations.relu)

        if use_fb:
            # self.upsample = layers.UpSampling2D(size=(us_factor, us_factor))
            self.conv_fb = layers.Conv2DTranspose(fb_channels, kernel_size=3, strides=(us_factor,us_factor), padding="same", use_bias=False)

        self.norm_fb = layers.BatchNormalization()
        self.conv_lat = layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False)

        # if use_fb2:
        #     self.conv_fb2 = layers.Conv2DTranspose(fb2_channels, kernel_size=3, strides=(us2_factor,us2_factor), padding="same", use_bias=False)
        #     self.norm_fb2 = layers.BatchNormalization()

        for t in range(self.times):
            setattr(self, f'norm1_{t}', layers.BatchNormalization())
            setattr(self, f'norm2_{t}', layers.BatchNormalization())
            setattr(self, f'norm3_{t}', layers.BatchNormalization())


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
            layers.BatchNormalization(),
            layers.Activation(activations.relu),
            layers.MaxPooling2D(pool_size=(2,2), padding="same"),
            layers.Conv2D(64, kernel_size=3, strides=(1,1), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(activations.relu)
        ]

        self.V1_Lam = layers.Lambda(lambda x: x, name="V1")

        self.nonlin = layers.Activation(activations.relu)

        self.V1_conv_fb1 = layers.Conv2DTranspose(64, kernel_size=3, strides=(1,1), padding="same", use_bias=False)
        self.V1_conv_fb2 = layers.Conv2DTranspose(64, kernel_size=3, strides=(4,4), padding="same", use_bias=False)
        self.V1_norm_fb = layers.BatchNormalization()
        self.V1_conv_lat = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False)

        self.V2 = RecurrentCNNBlock(128, times=1, use_fb=True, fb_channels=128, us_factor=2)
        self.mp1 = layers.MaxPooling2D(pool_size=(2,2), padding="same", name="V2")
        self.V4 = RecurrentCNNBlock(256, times=1, use_fb=True, fb_channels=256, us_factor=2)
        self.mp2 = layers.MaxPooling2D(pool_size=(2,2), padding="same", name="V4")
        self.IT = RecurrentCNNBlock(512, times=1, use_fb=False)

        self.IT_Lam = layers.Lambda(lambda x: x, name="IT")

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
            x = model.V1_Lam(x)

            x = self.V2(x, td_inp=self.V4.hidden_state)
            x = self.mp1(x)
            # x = v4_response + self.V4.conv_lat(v4_state) + self.V4.conv_fb(self.IT.hidden_state)
            # x = self.V4.norm_fb(x)
            # x = self.nonlin(x)
            # v4_state = x
            x = self.V4(x, td_inp=self.IT.hidden_state)
            x = self.mp2(x)
            x = self.IT(x)
            x = model.IT_Lam(x)

        #   Reset all hidden states to None here.
        self.V2.hidden_state = None
        self.V4.hidden_state = None
        self.IT.hidden_state = None

        for stage in self.decoder:
            x = stage(x)

        return x


def get_model_list():
    return ['sketch_model_4u-ep12']


def get_model(name):
    assert name == 'sketch_model_4u-ep12'

    model = RecurrentCNN(fb_loops=4)
    download_weights(
        bucket='brainscore-vision', folder_path='models/4u_model_submission/rcnn_cand_4u_weights_12ep',
        filename_version_sha=[
            ('.data-00000-of-00001', '697MOBz0KX7DPsunWjtLF96ylneRBNnb', 'todo'),
            ('.index', '2dTpZUFDq1QipHJthpC_3iFelB0QIvMK', 'todo'),
            ('checkpoint', '0Vhcz4Ou2olEbJ0d0xocEEp_F1CtK2Fc', 'todo')],
        save_directory=Path(__file__).parent / 'models' / 'rcnn_cand_4u_weights_10ep')
    model.load_weights(os.path.join(os.path.dirname(__file__), 'rcnn_cand_4u_weights_12ep/'))

    inputs = keras.Input(shape=(32, 32, 3))
    x = inputs
    #   Perform call here.
    for stage in model.V1:
        x = stage(x)

    #   First pass to save initial states.
    v1_response = x
    v1_state = x
    x = model.V2(x)
    x = model.mp1(x)
    x = model.V4(x)
    # v4_response = x
    # v4_state = x
    x = model.mp2(x)
    x = model.IT(x)


    for t in range(model.fb_loops):
        x = v1_response + model.V1_conv_lat(v1_state) + model.V1_conv_fb1(model.V2.hidden_state) + model.V1_conv_fb2(model.IT.hidden_state)
        x = model.V1_norm_fb(x)
        x = model.nonlin(x)
        v1_state = x
        x = model.V1_Lam(x)

        x = model.V2(x, td_inp=model.V4.hidden_state)
        x = model.mp1(x)
        # x = v4_response + self.V4.conv_lat(v4_state) + self.V4.conv_fb(self.IT.hidden_state)
        # x = self.V4.norm_fb(x)
        # x = self.nonlin(x)
        # v4_state = x
        x = model.V4(x, td_inp=model.IT.hidden_state)
        x = model.mp2(x)
        x = model.IT(x)
        x = model.IT_Lam(x)

    #   Reset all hidden states to None here.
    model.V2.hidden_state = None
    model.V4.hidden_state = None
    model.IT.hidden_state = None

    for stage in model.decoder:
        x = stage(x)

    outputs = x
    #   End call.

    model = keras.Model(inputs=inputs, outputs=outputs)
    #   Might need to set image size to 224 - seems to run just fine.
    model_preprocessing = keras.applications.vgg16.preprocess_input
    load_preprocess = lambda image_filepaths: model_preprocessing(load_images(image_filepaths, image_size=224))
    wrapper = KerasWrapper(model, load_preprocess)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'sketch_model_4u-ep12'
    #   Technically, the end activation for V1 would be v1_state.  Perhaps I can add an Identity matrix after v1_state
    #   is updated so that it is detected as a layer in the model.
    return ['V1', 'V2', 'V4', 'IT']


if __name__ == '__main__':
    check_models.check_base_models(__name__)

