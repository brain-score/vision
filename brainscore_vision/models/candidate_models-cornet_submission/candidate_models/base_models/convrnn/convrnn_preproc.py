
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ImageNet preprocessing for ResNet."""

import tensorflow as tf
import numpy as np

def color_normalize(image):
    image = tf.cast(image, tf.float32) / 255
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std
    return image

def get_resize_scale(height, width, crop_size=224, smallest_side=256):
    """
    Get the resize scale so that the shortest side is `smallest_side`
    """
    smallest_side = tf.convert_to_tensor(value=smallest_side, dtype=tf.int32)

    height = tf.cast(height, dtype=tf.float32)
    width = tf.cast(width, dtype=tf.float32)
    smallest_side = tf.cast(smallest_side, dtype=tf.float32)

    scale = tf.cond(
            pred=tf.greater(height, width),
            true_fn=lambda: smallest_side / width,
            false_fn=lambda: smallest_side / height)
    return scale

def resize_cast_to_uint8(image, crop_size=224):
    image = tf.cast(
            tf.image.resize(
                [image],
                [crop_size, crop_size], method=tf.image.ResizeMethod.BILINEAR)[0],
            dtype=tf.uint8)
    image.set_shape([crop_size, crop_size, 3])
    return image

def central_crop_from_jpg(image_string, crop_size=224, smallest_side=256):
    """
    Resize the image to make its smallest side to be 256;
    then get the central 224 crop
    """
    shape = tf.image.extract_jpeg_shape(image_string)
    scale = get_resize_scale(shape[0], shape[1], crop_size=crop_size, smallest_side=smallest_side)
    cp_height = tf.cast(crop_size / scale, tf.int32)
    cp_width = tf.cast(crop_size / scale, tf.int32)
    cp_begin_x = tf.cast((shape[0] - cp_height) / 2, tf.int32)
    cp_begin_y = tf.cast((shape[1] - cp_width) / 2, tf.int32)
    bbox = tf.stack([
            cp_begin_x, cp_begin_y, \
            cp_height, cp_width])
    crop_image = tf.image.decode_and_crop_jpeg(
            image_string,
            bbox,
            channels=3)
    image = resize_cast_to_uint8(crop_image, crop_size=crop_size)

    return image

def preprocess_for_eval(image_string, resize=None, crop_size=224, smallest_side=256):
    image = central_crop_from_jpg(image_string, crop_size=crop_size, smallest_side=smallest_side)

    image = color_normalize(image)

    if resize is not None:
        image = tf.image.resize(image, [resize, resize])
    return image
