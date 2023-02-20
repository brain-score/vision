import logging

import pytest
from pytest import approx

from brainscore.benchmarks.imagenet import Imagenet2012
from candidate_models.model_commitments import brain_translated_pool

_logger = logging.getLogger(__name__)


@pytest.mark.memory_intense
@pytest.mark.requires_gpu
@pytest.mark.slow
class TestImagenet:
    @pytest.mark.parametrize(['model', 'expected_top1', 'allowed_deviation'], [
        # pytorch: from https://pytorch.org/docs/stable/torchvision/models.html
        ('alexnet', 1 - .4345, .07),
        ('squeezenet1_0', 1 - .4190, .07),
        ('squeezenet1_1', 1 - .4181, .07),
        ('resnet-18', 1 - .3024, .07),
        ('resnet-34', 1 - .2670, .07),
        ('resnet-50-pytorch', 1 - .2385, .07),
        ('resnet-50-robust', .5332, .07),  # computed manually, as no score was given with
        # keras: from https://keras.io/applications/#documentation-for-individual-models
        ('xception', .790, .07),
        ('vgg-16', .713, .07),
        ('vgg-19', .713, .07),
        ('densenet-121', .750, .07),
        ('densenet-169', .762, .07),
        ('densenet-201', .773, .07),
        # tf-slim: from
        # https://github.com/tensorflow/models/tree/b3158fb0183809400e9e7f8092dd541201b1c4d4/research/slim#pre-trained-models
        ('inception_v1', .698, .07),
        ('inception_v2', .739, .07),
        ('inception_v3', .780, .07),
        ('inception_v4', .802, .07),
        ('inception_resnet_v2', .804, .07),
        ('resnet-50_v1', .752, .07),
        ('resnet-101_v1', .764, .07),
        ('resnet-152_v1', .768, .07),
        ('resnet-50_v2', .756, .07),
        ('resnet-101_v2', .770, .07),
        ('resnet-152_v2', .778, .07),
        ('nasnet_mobile', .740, .07),
        ('nasnet_large', .827, .07),
        ('pnasnet_large', .829, .07),
        ('mobilenet_v1_1.0_224', 0.709, .07),
        ('mobilenet_v1_1.0_192', 0.7, .07),
        ('mobilenet_v1_1.0_160', 0.68, .07),
        ('mobilenet_v1_1.0_128', 0.652, .07),
        ('mobilenet_v1_0.75_224', 0.684, .07),
        ('mobilenet_v1_0.75_192', 0.672, .07),
        ('mobilenet_v1_0.75_160', 0.653, .07),
        ('mobilenet_v1_0.75_128', 0.621, .07),
        ('mobilenet_v1_0.5_224', 0.633, .07),
        ('mobilenet_v1_0.5_192', 0.617, .07),
        ('mobilenet_v1_0.5_160', 0.591, .07),
        ('mobilenet_v1_0.5_128', 0.563, .07),
        ('mobilenet_v1_0.25_224', 0.498, .07),
        ('mobilenet_v1_0.25_192', 0.477, .07),
        ('mobilenet_v1_0.25_160', 0.455, .07),
        ('mobilenet_v1_0.25_128', 0.415, .07),
        ('mobilenet_v2_1.4_224', 0.75, .07),
        ('mobilenet_v2_1.3_224', 0.744, .07),
        ('mobilenet_v2_1.0_224', 0.718, .07),
        ('mobilenet_v2_1.0_192', 0.707, .07),
        ('mobilenet_v2_1.0_160', 0.688, .07),
        ('mobilenet_v2_1.0_128', 0.653, .07),
        ('mobilenet_v2_1.0_96', 0.603, .07),
        ('mobilenet_v2_0.75_224', 0.698, .07),
        ('mobilenet_v2_0.75_192', 0.687, .07),
        ('mobilenet_v2_0.75_160', 0.664, .07),
        ('mobilenet_v2_0.75_128', 0.632, .07),
        ('mobilenet_v2_0.75_96', 0.588, .07),
        ('mobilenet_v2_0.5_224', 0.654, .07),
        ('mobilenet_v2_0.5_192', 0.639, .07),
        ('mobilenet_v2_0.5_160', 0.61, .07),
        ('mobilenet_v2_0.5_128', 0.577, .07),
        ('mobilenet_v2_0.5_96', 0.512, .07),
        ('mobilenet_v2_0.35_224', 0.603, .07),
        ('mobilenet_v2_0.35_192', 0.582, .07),
        ('mobilenet_v2_0.35_160', 0.557, .07),
        ('mobilenet_v2_0.35_128', 0.508, .07),
        ('mobilenet_v2_0.35_96', 0.455, .07),
        # bagnet: own runs
        ('bagnet9', .2635, .01),
        ('bagnet17', .46, .01),
        ('bagnet33', .58924, .01),
        # # ConvRNN: from https://arxiv.org/abs/1807.00053, page 6
        ('convrnn_224', 0.729, .04),
        # resnet stylized ImageNet: from https://openreview.net/pdf?id=Bygh9j09KX, Table 2
        ('resnet50-SIN', .6018, .07),
        ('resnet50-SIN_IN', .7459, .07),
        ('resnet50-SIN_IN_IN', .7672, .07),
        # wsl: from https://github.com/facebookresearch/WSL-Images/tree/c4dac640995f66db893410d6d4356d49a9d3dcc0
        ('resnext101_32x8d_wsl', .822, .07),
        ('resnext101_32x16d_wsl', .842, .07),
        ('resnext101_32x32d_wsl', .851, .07),
        ('resnext101_32x48d_wsl', .854, .07),
        # FixRes: from https://arxiv.org/pdf/1906.06423.pdf, Table 8
        ('fixres_resnext101_32x48d_wsl', .863, .07),
    ])
    def test_top1(self, model, expected_top1, allowed_deviation):
        # clear tf graph
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        import keras
        keras.backend.clear_session()
        # run
        _model = brain_translated_pool[model]
        benchmark = Imagenet2012()
        score = benchmark(_model)
        accuracy = score.sel(aggregation='center')
        _logger.debug(f"{model} ImageNet2012-top1 -> {accuracy} (expected {expected_top1})")
        assert accuracy == approx(expected_top1, abs=allowed_deviation)
