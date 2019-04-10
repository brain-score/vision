import os

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
        _objectome = self.get_objectome()
        fitting_objectome, testing_objectome = _objectome[~ _objectome['use']], _objectome[_objectome['use']]

        # features
        feature_responses = pd.read_pickle(os.path.join(os.path.dirname(__file__),
                                                        f'identifier={model},stimuli_identifier=objectome-240.pkl'))
        feature_responses = feature_responses['data']
        feature_responses['image_id'] = 'stimulus_path', [os.path.splitext(os.path.basename(path))[0]
                                                          for path in feature_responses['stimulus_path'].values]
        feature_responses = feature_responses.stack(presentation=['stimulus_path'])
        expected_images = set(fitting_objectome['id'].values) | set(testing_objectome['id'].values)
        assert expected_images.issuperset(set(feature_responses['image_id'].values))
        assert len(np.unique(feature_responses['layer'])) == 1  # only penultimate layer
        feature_responses = feature_responses.transpose('presentation', 'neuroid')
        fitting_features = self.separate_and_annotate_features(feature_responses, fitting_objectome)
        testing_features = self.separate_and_annotate_features(feature_responses, testing_objectome)
        # metric
        testing_objectome = self.to_xarray(testing_objectome)
        i2n = I2n()
        score = i2n(fitting_features, testing_features, testing_objectome)
        score = score.sel(aggregation='center')
        assert score == approx(expected_score, abs=0.005), f"expected {expected_score}, but got {score}"

    def get_objectome(self):
        packaged_filepath = os.path.join(os.path.dirname(__file__), 'objectome240.pkl')
        if not os.path.isfile(packaged_filepath):
            # repackage subsampled
            objectome = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'objectome24s100_humanpool.pkl'))
            subsample = pd.read_pickle(
                os.path.join(os.path.dirname(__file__), 'objectome24s100_imgsubsampled240_pandas.pkl'))
            objectome['use'] = objectome['id'].isin(subsample.values[:, 0])
            objectome.to_pickle(packaged_filepath)
        else:
            objectome = pd.read_pickle(packaged_filepath)
        objectome['correct'] = objectome['choice'] == objectome['sample_obj']
        objectome['label'] = objectome['sample_obj']
        return objectome

    def separate_and_annotate_features(self, feature_responses, objectome):
        id_rows = {row['id']: row for _, row in objectome[['id', 'label']].drop_duplicates().iterrows()}
        image_ids = xr.DataArray(np.zeros(len(id_rows)), coords={'image_id': list(id_rows.keys())},
                                 dims=['image_id']).stack(presentation=['image_id'])
        feature_responses = subset(feature_responses, image_ids)
        labels = [id_rows[image_id]['label'] for image_id in feature_responses['image_id'].values]
        feature_responses['label'] = 'presentation', labels
        return feature_responses

    def to_xarray(self, objectome):
        columns = objectome.columns
        objectome = xr.DataArray(objectome['choice'],
                                 coords={column: ('presentation', objectome[column]) for column in columns},
                                 dims=['presentation'])
        objectome = objectome.rename({'id': 'image_id'})
        objectome['truth'] = objectome['label']
        objectome = objectome.set_index(presentation=[col if col != 'id' else 'image_id' for col in columns])
        return objectome
