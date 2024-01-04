from model_tools.activations.keras import load_images, KerasWrapper
import keras.applications
from model_tools.check_submission import check_models

# This is an example implementation for submitting vgg-16 as a keras model to brain-score
# If you use keras, don't forget to add it and its dependencies to the setup.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers
import os

import numpy as np
from typeguard import typechecked
import logging
from typing import Union, Callable, List
# from tensorflow.python.keras.engine import keras_tensor

Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

Initializer = Union[None, dict, str, Callable, tf.keras.initializers.Initializer]
Regularizer = Union[None, dict, str, Callable, tf.keras.regularizers.Regularizer]
Constraint = Union[None, dict, str, Callable, tf.keras.constraints.Constraint]
Activation = Union[None, str, Callable]
Optimizer = Union[tf.keras.optimizers.Optimizer, str]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
    # keras_tensor.KerasTensor,
]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]


class GroupNormalization(tf.keras.layers.Layer):
    @typechecked
    def __init__(
        self,
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: Initializer = "zeros",
        gamma_initializer: Initializer = "ones",
        beta_regularizer: Regularizer = None,
        gamma_regularizer: Regularizer = None,
        beta_constraint: Constraint = None,
        gamma_constraint: Constraint = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape


class InstanceNormalization(GroupNormalization):
    def __init__(self, **kwargs):
        if "groups" in kwargs:
            logging.warning("The given value for groups will be overwritten.")

        kwargs["groups"] = -1
        super().__init__(**kwargs)




#   Based on CORnet-S.
#   Source: https://github.com/dicarlolab/CORnet
class RecurrentCNNBlock(layers.Layer):

    scale = 2

    def __init__(self, out_channels, times=1, use_fb=False, fb_channels=None, us_factor=None, pool_factor=None):
        super(RecurrentCNNBlock, self).__init__()

        self.hidden_state = None

        self.times = times
        self.use_fb = use_fb
        self.conv_input = layers.Conv2D(out_channels, kernel_size=1, padding="same", use_bias=False)
        self.skip = layers.Conv2D(out_channels, kernel_size=1, strides=(1,1), padding="same", use_bias=False)
        self.norm_skip = InstanceNormalization()

        self.conv1 = layers.Conv2D(out_channels * self.scale, kernel_size=1, padding="same", use_bias=False)
        self.nonlin1 = layers.Activation(activations.relu)
        self.conv2 = layers.Conv2D(out_channels * self.scale, kernel_size=3, strides=(1,1), padding="same", use_bias=False)
        self.nonlin2 = layers.Activation(activations.relu)
        self.conv3 = layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False)
        self.nonlin3 = layers.Activation(activations.relu)

        if use_fb:
            self.conv_fb = layers.Conv2DTranspose(fb_channels, kernel_size=3, strides=(us_factor,us_factor), padding="same", use_bias=False)

        if pool_factor is not None:
            self.conv_ff_skip = layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False)
            self.skip_pool = layers.MaxPooling2D(pool_size=(pool_factor, pool_factor), strides=(pool_factor, pool_factor), padding="same")

        self.norm_fb = InstanceNormalization()
        self.conv_lat = layers.Conv2D(out_channels, kernel_size=3, padding="same", use_bias=False)
        self.Add = layers.Add()

        for t in range(self.times):
            setattr(self, f'norm1_{t}', InstanceNormalization())
            setattr(self, f'norm2_{t}', InstanceNormalization())
            setattr(self, f'norm3_{t}', InstanceNormalization())

        self.Noise = layers.GaussianNoise(0.1)


    #   Recurrent CNN call.
    #   Sourced:  https://github.com/cjspoerer/rcnn-sat
    def call(self, inp, td_inp=None, skip_inp=None, fb=False, fb2=False, training=False):
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
            x = self.Add([x, skip])

            # if t != self.times-1:
            x = self.nonlin3(x)

        #   First step of recurrence so no top-down or lateral input yet.
        if self.hidden_state is None:
            # x = self.nonlin3(x)
            x = self.Noise(x)
            self.hidden_state = x
        else:
            x = self.Add([x, self.conv_lat(self.hidden_state)])
            if td_inp is not None:
                x = self.Add([x, self.conv_fb(td_inp)])

            if skip_inp is not None:
                x = self.Add([x, self.skip_pool(self.conv_ff_skip(skip_inp))])

            x = self.norm_fb(x)
            x = self.nonlin3(x)
            x = self.Noise(x)
            self.hidden_state = x

        return x


class RecurrentCNN(keras.Model):
    def __init__(self, fb_loops=4):
        super(RecurrentCNN, self).__init__()

        self.fb_loops = fb_loops

        self.V1 = [
            layers.Conv2D(64, kernel_size=7, strides=(1,1), padding="same", use_bias=False),
            InstanceNormalization(),
            layers.Activation(activations.relu),
            layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same"),
            layers.Conv2D(64, kernel_size=3, strides=(1,1), padding="same", use_bias=False),
            InstanceNormalization(),
            layers.Activation(activations.relu),
            layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
        ]

        self.nonlin = layers.Activation(activations.relu)
        self.noise = layers.GaussianNoise(0.1)

        self.V1_Add = layers.Add()
        self.V1_conv_fb1 = layers.Conv2D(64, kernel_size=3, strides=(1,1), padding="same", use_bias=False)
        self.V1_conv_fb2 = layers.Conv2DTranspose(64, kernel_size=3, strides=(2,2), padding="same", use_bias=False)
        self.V1_conv_fb3 = layers.Conv2DTranspose(64, kernel_size=3, strides=(4,4), padding="same", use_bias=False)
        self.V1_norm_fb = InstanceNormalization()
        self.V1_conv_lat = layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False)

        self.V2 = RecurrentCNNBlock(128, times=2, use_fb=True, fb_channels=128, us_factor=2)
        self.mp1 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
        self.V4 = RecurrentCNNBlock(256, times=2, use_fb=True, fb_channels=256, us_factor=2, pool_factor=2)
        self.mp2 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
        self.IT = RecurrentCNNBlock(512, times=2, use_fb=False)

        self.V1_Lam = layers.Lambda(lambda x: x, name="V1")
        self.V2_Lam = layers.Lambda(lambda x: x, name="V2")
        self.V4_Lam = layers.Lambda(lambda x: x, name="V4")
        self.IT_Lam = layers.Lambda(lambda x: x, name="IT")

        self.decoder = [
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='softmax')
        ]


    #   Recurrent CNN call.
    def call(self, x, training=None, mask=None):

        for stage in self.V1:
            x = stage(x)

        #   First pass to save initial states.
        v1_response = x
        v1_state = x
        x = self.noise(x)
        x = self.V2(x)
        x = self.mp1(x)
        x = self.V4(x)
        x = self.mp2(x)
        self.IT(x)

        for t in range(self.fb_loops-1):
            x = self.V1_Add([
                v1_response, self.V1_conv_lat(v1_state), self.V1_conv_fb1(self.V2.hidden_state),
                self.V1_conv_fb2(self.V4.hidden_state)
            ])

            x = self.V1_norm_fb(x)
            x = self.nonlin(x)
            x = self.noise(x)
            v1_state = x

            x = self.V2(x, td_inp=self.V4.hidden_state)
            x = self.mp1(x)

            x = self.V4(x, td_inp=self.IT.hidden_state, skip_inp=v1_state)
            x = self.mp2(x)

            self.IT(x)

        #   For final pass, include lambda identity layers in hopes that these final values are used rather than ones earlier.
        x = self.V1_Add([
            v1_response, self.V1_conv_lat(v1_state), self.V1_conv_fb1(self.V2.hidden_state),
            self.V1_conv_fb2(self.V4.hidden_state)
        ])

        x = self.V1_norm_fb(x)
        x = self.nonlin(x)
        x = self.noise(x)
        v1_state = x
        x = self.V1_Lam(x)

        x = self.V2(x, td_inp=self.V4.hidden_state)
        x = self.mp1(x)
        x = self.V2_Lam(x)

        x = self.V4(x, td_inp=self.IT.hidden_state, skip_inp=v1_state)
        x = self.mp2(x)
        x = self.V4_Lam(x)

        x = self.IT(x)
        x = self.IT_Lam(x)

        #   Reset all hidden states to None here.
        self.V2.hidden_state = None
        self.V4.hidden_state = None
        self.IT.hidden_state = None

        for stage in self.decoder:
            x = stage(x)

        return x


def get_model_list():
    return ['sketch_model_8a-ep24']


def get_model(name):
    assert name == 'sketch_model_8a-ep24'

    model = RecurrentCNN(fb_loops=4)
    model.load_weights(os.path.join(os.path.dirname(__file__), 'rcnn_cand_8a_weights_24ep/'))

    inputs = keras.Input(shape=(224, 224, 3))
    x = inputs

    #   Perform call here.
    for stage in model.V1:
        x = stage(x)

    #   First pass to save initial states.
    v1_response = x
    v1_state = x
    x = model.noise(x)
    x = model.V2(x)
    x = model.mp1(x)
    x = model.V4(x)
    x = model.mp2(x)
    x = model.IT(x)

    for t in range(model.fb_loops-1):
        x = model.V1_Add([
                v1_response, model.V1_conv_lat(v1_state), model.V1_conv_fb1(model.V2.hidden_state),
                model.V1_conv_fb2(model.V4.hidden_state)
        ])

        x = model.V1_norm_fb(x)
        x = model.nonlin(x)
        x = model.noise(x)
        v1_state = x

        x = model.V2(x, td_inp=model.V4.hidden_state)
        x = model.mp1(x)

        x = model.V4(x, td_inp=model.IT.hidden_state, skip_inp=v1_state)
        x = model.mp2(x)

        model.IT(x)

    #   For final pass, include lambda identity layers in hopes that these final values are used rather than ones earlier.
    x = model.V1_Add([
            v1_response, model.V1_conv_lat(v1_state), model.V1_conv_fb1(model.V2.hidden_state),
            model.V1_conv_fb2(model.V4.hidden_state)
    ])

    x = model.V1_norm_fb(x)
    x = model.nonlin(x)
    x = model.noise(x)
    v1_state = x
    x = model.V1_Lam(x)

    x = model.V2(x, td_inp=model.V4.hidden_state)
    x = model.mp1(x)
    x = model.V2_Lam(x)

    x = model.V4(x, td_inp=model.IT.hidden_state, skip_inp=v1_state)
    x = model.mp2(x)
    x = model.V4_Lam(x)

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
    assert name == 'sketch_model_8a-ep24'
    #   Technically, the end activation for V1 would be v1_state.  Perhaps I can add an Identity matrix after v1_state
    #   is updated so that it is detected as a layer in the model.
    return ['V1', 'V2', 'V4', 'IT']


if __name__ == '__main__':
    check_models.check_base_models(__name__)

