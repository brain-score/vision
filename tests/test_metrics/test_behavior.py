import os
import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import approx

from brainscore.metrics.behavior import I2n
from brainscore.metrics.transformations import subset


class TestI2N:
    @pytest.mark.parametrize(['model', 'expected_score'],
                             [
                                 ('alexnet', .253),
                                 ('resnet34', .37787),
                                 ('resnet18', .3638),
                             ])
    def test_model(self, model, expected_score):
        # assembly
        fitting_objectome, testing_objectome = self.get_objectome('partial_trials'), self.get_objectome('full_trials')

        # features
        feature_responses = pd.read_pickle(os.path.join(os.path.dirname(__file__),
                                                        f'identifier={model},stimuli_identifier=objectome-240.pkl'))
        feature_responses = feature_responses['data']
        feature_responses['image_id'] = 'stimulus_path', [os.path.splitext(os.path.basename(path))[0]
                                                          for path in feature_responses['stimulus_path'].values]
        feature_responses = feature_responses.stack(presentation=['stimulus_path'])
        expected_images = set(fitting_objectome['image_id'].values) | set(testing_objectome['image_id'].values)
        assert expected_images.issuperset(set(feature_responses['image_id'].values))
        assert len(np.unique(feature_responses['layer'])) == 1  # only penultimate layer
        feature_responses = feature_responses.transpose('presentation', 'neuroid')
        fitting_features = self.separate_and_annotate_features(feature_responses, fitting_objectome)
        testing_features = self.separate_and_annotate_features(feature_responses, testing_objectome)
        # metric
        i2n = I2n()
        score = i2n(fitting_features, testing_features, testing_objectome)
        score = score.sel(aggregation='center')
        assert score == approx(expected_score, abs=0.005), f"expected {expected_score}, but got {score}"

    def get_objectome(self, subtype):
        with open(f'/braintree/home/msch/brainio_contrib/mkgu_packaging/dicarlo/dicarlo.Rajalingham2018.{subtype}.pkl',
                  'rb') as f:
            objectome = pickle.load(f)
        return objectome

    def separate_and_annotate_features(self, feature_responses, objectome):
        image_labels = {image_id: label for image_id, label in
                        zip(objectome['image_id'].values, objectome['truth'].values)}
        image_ids = xr.DataArray(np.zeros(len(image_labels)), coords={'image_id': list(image_labels.keys())},
                                 dims=['image_id']).stack(presentation=['image_id'])
        feature_responses = subset(feature_responses, image_ids)
        labels = [image_labels[image_id] for image_id in feature_responses['image_id'].values]
        feature_responses['label'] = 'presentation', labels
        return feature_responses
