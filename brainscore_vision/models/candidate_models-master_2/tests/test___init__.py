import functools
from typing import Union

import numpy as np
import pytest
from pytest import approx

from brainscore import score_model
from brainscore.utils import LazyLoad
from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import brain_translated_pool
from model_tools.activations import PytorchWrapper
from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation import LayerMappedModel, TemporalIgnore


@pytest.mark.private_access
class TestPreselectedLayer:
    def layer_candidate(self, model_name, layer, region, pca_components: Union[None, int] = 1000):
        def load(model_name=model_name, layer=layer, region=region, pca_components=pca_components):
            activations_model = base_model_pool[model_name]
            if pca_components:
                LayerPCA.hook(activations_model, n_components=pca_components)
                activations_model.identifier += "-pca_1000"
            model = LayerMappedModel(f"{model_name}-{layer}", activations_model=activations_model, visual_degrees=8,
                                     region_layer_map={region: layer})
            model = TemporalIgnore(model)
            return model

        return LazyLoad(load)  # lazy-load to avoid loading all models right away

    @pytest.mark.memory_intense
    def test_alexnet_conv2_V4(self):
        model = self.layer_candidate('alexnet', layer='features.5', region='V4', pca_components=1000)
        score = score_model(model_identifier='alexnet-f5-pca_1000', model=model,
                            benchmark_identifier='dicarlo.MajajHong2015.V4-pls')
        assert score.raw.sel(aggregation='center').max() == approx(0.633703, abs=0.005)

    @pytest.mark.memory_intense
    def test_alexnet_conv5_V4(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='V4', pca_components=1000)
        score = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                            benchmark_identifier='dicarlo.MajajHong2015.V4-pls')
        assert score.raw.sel(aggregation='center') == approx(0.490769, abs=0.005)

    @pytest.mark.memory_intense
    def test_alexnet_conv5_IT(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='IT', pca_components=1000)
        score = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                            benchmark_identifier='dicarlo.MajajHong2015.IT-pls')
        assert score.raw.sel(aggregation='center') == approx(0.590345, abs=0.005)

    @pytest.mark.memory_intense
    def test_alexnet_conv3_IT_mask(self):
        model = self.layer_candidate('alexnet', layer='features.6', region='IT', pca_components=None)
        np.random.seed(123)
        score = score_model(model_identifier='alexnet-f6', model=model,
                            benchmark_identifier='dicarlo.MajajHong2015.IT-mask')
        assert score.raw.sel(aggregation='center') == approx(0.607037, abs=0.005)

    @pytest.mark.memory_intense
    def test_repeat_same_result(self):
        model = self.layer_candidate('alexnet', layer='features.12', region='IT', pca_components=1000)
        score1 = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                             benchmark_identifier='dicarlo.MajajHong2015.IT-pls')
        score2 = score_model(model_identifier='alexnet-f12-pca_1000', model=model,
                             benchmark_identifier='dicarlo.MajajHong2015.IT-pls')
        assert (score1 == score2).all()

    def test_newmodel_pytorch(self):
        import torch
        from torch import nn
        from model_tools.activations.pytorch import load_preprocess_images

        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
                self.relu1 = torch.nn.ReLU()
                linear_input_size = np.power((224 - 3 + 2 * 0) / 1 + 1, 2) * 2
                self.linear = torch.nn.Linear(int(linear_input_size), 1000)
                self.relu2 = torch.nn.ReLU()  # can't get named ReLU output otherwise

                # init weights for reproducibility
                self.conv1.weight.data.fill_(0.01)
                self.conv1.bias.data.fill_(0.01)
                self.linear.weight.data.fill_(0.01)
                self.linear.bias.data.fill_(0.01)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = x.view(x.size(0), -1)
                x = self.linear(x)
                x = self.relu2(x)
                return x

        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        model_id = 'new_pytorch'
        activations_model = PytorchWrapper(model=MyModel(), preprocessing=preprocessing, identifier=model_id)
        layer = 'relu2'
        candidate = LayerMappedModel(f"{model_id}-{layer}", activations_model=activations_model, visual_degrees=8,
                                     region_layer_map={'IT': layer})
        candidate = TemporalIgnore(candidate)

        ceiled_score = score_model(model_identifier=model_id, model=candidate,
                                   benchmark_identifier='dicarlo.MajajHong2015.IT-pls')
        score = ceiled_score.raw
        assert score.sel(aggregation='center') == approx(.0820823, abs=.01)


@pytest.mark.private_access
@pytest.mark.memory_intense
class TestBrainTranslated:
    @pytest.mark.parametrize(['model_identifier', 'expected_score', 'attach_hook'], [
        ('alexnet', approx(.59033, abs=.005), True),
        ('resnet18-supervised', approx(.596013, abs=.005), False),
        ('resnet18-local_aggregation', approx(.60152, abs=.005), False),
        ('resnet18-autoencoder', approx(.373528, abs=.005), False),
        ('CORnet-S', approx(.600, abs=.02), False),
        ('VOneCORnet-S', approx(.610, abs=.02), False),
    ])
    def test_MajajHong2015ITpls(self, model_identifier, expected_score, attach_hook):
        model = brain_translated_pool[model_identifier]
        if attach_hook:
            activations_model = model.layer_model._layer_model.activations_model
            LayerPCA.hook(activations_model, n_components=1000)
            identifier = activations_model.identifier + "-pca_1000"
            activations_model.identifier = identifier
        score = score_model(model_identifier, 'dicarlo.MajajHong2015.IT-pls', model=model)
        assert score.raw.sel(aggregation='center') == expected_score

    @pytest.mark.parametrize(['model_identifier', 'expected_score'], [
        ('CORnet-S', approx(.240888, abs=.002)),
        ('CORnet-R2', approx(.230859, abs=.002)),
        ('alexnet', None),
        ('VOneCORnet-S', approx(.23107, abs=.01)),
    ])
    def test_candidate_Kar2019OST(self, model_identifier, expected_score):
        model = brain_translated_pool[model_identifier]
        score = score_model(model_identifier=model_identifier, model=model, benchmark_identifier='dicarlo.Kar2019-ost')
        if expected_score is not None:
            assert score.raw.sel(aggregation='center') == expected_score
        else:
            assert np.isnan(score.raw.sel(aggregation='center'))

    @pytest.mark.parametrize(['model_identifier', 'expected_score'], [
        ('CORnet-S',  approx(.382, abs=.005)),
        ('alexnet', approx(.253, abs=.005)),
        ('VOneCORnet-S', approx(.356, abs=.02)),
        ('voneresnet-50', approx(.371, abs=.02)),
        ('voneresnet-50-robust', approx(.386, abs=.02)),
    ])
    def test_Rajalingham2018i2n(self, model_identifier, expected_score):
        model = brain_translated_pool[model_identifier]
        score = score_model(model_identifier=model_identifier, model=model,
                            benchmark_identifier='dicarlo.Rajalingham2018-i2n')
        assert score.raw.sel(aggregation='center') == expected_score

    @pytest.mark.parametrize('model_identifier', [
        'CORnet-S',
        'alexnet',
    ])
    def test_brain_translated_pool_reload(self, model_identifier):
        activations_model = brain_translated_pool[model_identifier].content.activations_model
        LayerPCA.hook(activations_model, n_components=1000)
        assert len(activations_model._extractor._batch_activations_hooks) == 1
        activations_model = brain_translated_pool[model_identifier].content.activations_model
        assert len(activations_model._extractor._stimulus_set_hooks) == 0
        assert len(activations_model._extractor._batch_activations_hooks) == 0
        LayerPCA.hook(activations_model, n_components=1000)
        assert len(activations_model._extractor._batch_activations_hooks) == 1

    def test_base_model_pool_reload(self):
        activations_model = base_model_pool['alexnet']
        LayerPCA.hook(activations_model, n_components=1000)
        assert len(activations_model._extractor._batch_activations_hooks) == 1
        activations_model = base_model_pool['alexnet']
        assert len(activations_model._extractor._stimulus_set_hooks) == 0
        assert len(activations_model._extractor._batch_activations_hooks) == 0
        LayerPCA.hook(activations_model, n_components=1000)
        assert len(activations_model._extractor._batch_activations_hooks) == 1
