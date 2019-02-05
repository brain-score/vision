import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import approx

from brainio_base.assemblies import DataAssembly
from brainscore.metrics.behavior import I2n
from brainscore.metrics.transformations import subset


class TestI2N:
    def test_halves_match_precomputed(self):
        i2n = I2n()

        expected = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'metrics240.pkl'))['I2_dprime_C']
        meta = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'meta.pkl'))
        sel = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'sel240.pkl'))
        meta = meta.loc[sel]
        presentation_coords = {column.replace('id', 'image_id'): ('presentation', meta[column].values)
                               for column in meta.columns}
        expected_halfs = []
        for split in expected[0]:
            choices = [meta['obj'].values[np.where(np.isnan(split[:, choice_index]))[0][0]]
                       for choice_index in range(split.shape[1])]
            split = DataAssembly(split, coords={**presentation_coords, **{'choice': choices}},
                                 dims=['presentation', 'choice'])
            expected_halfs.append(split)

        objectome = self.get_objectome()
        objectome = self.to_xarray(objectome)
        objectome = objectome.sel(use=True)
        actual_halfs = []
        split_indices = np.arange(len(objectome))
        np.random.seed(0)
        np.random.shuffle(split_indices)
        split_indices = np.array_split(split_indices, 2)
        for split_idx in split_indices:
            split = objectome[split_idx]
            response_matrix = i2n.build_response_matrix_from_responses(split)
            response_matrix = i2n.normalize_response_matrix(response_matrix)
            actual_halfs.append(response_matrix)

        expected_halfs_correlation = i2n.correlate(*expected_halfs)
        actual_halfs_correlation = i2n.correlate(*actual_halfs)
        actual_expected_correlation1 = i2n.correlate(actual_halfs[0], expected_halfs[0])
        actual_expected_correlation2 = i2n.correlate(actual_halfs[1], expected_halfs[1])

        assert actual_expected_correlation1 == approx(.71, abs=.005)
        assert actual_halfs_correlation == approx(expected_halfs_correlation, abs=0.02)
        assert actual_expected_correlation2 == approx(actual_expected_correlation1, abs=0.02)

    @pytest.mark.parametrize(['model', 'expected_score'],
                             [
                                 ('alexnet', .245),
                                 ('resnet34', .378),
                                 ('resnet18', .364),
                                 ('squeezenet1_0', .180),
                                 ('squeezenet1_1', .201),
                             ])
    def test_model(self, model, expected_score):
        objectome = self.get_objectome()
        id_rows = {row['id']: row for _, row in objectome[['id', 'label', 'use']].drop_duplicates().iterrows()}
        feature_responses = pd.read_pickle(os.path.join(os.path.dirname(__file__),
                                                        f'identifier={model},stimuli_identifier=objectome-240.pkl'))
        feature_responses = feature_responses['data']
        feature_responses['image_id'] = 'stimulus_path', [os.path.splitext(os.path.basename(path))[0]
                                                          for path in feature_responses['stimulus_path'].values]
        feature_responses = feature_responses.stack(presentation=['stimulus_path'])
        assert set(objectome['id'].values).issuperset(set(feature_responses['image_id'].values))
        assert len(np.unique(feature_responses['layer'])) == 1  # only penultimate layer
        image_ids = xr.DataArray(np.zeros(len(id_rows)), coords={'image_id': list(id_rows.keys())},
                                 dims=['image_id']).stack(presentation=['image_id'])
        feature_responses = subset(feature_responses, image_ids)
        labels = [id_rows[image_id]['label'] for image_id in feature_responses['image_id'].values]
        feature_responses['label'] = 'presentation', labels
        use = [id_rows[image_id]['use'] for image_id in feature_responses['image_id'].values]
        feature_responses['use'] = 'presentation', use
        feature_responses = feature_responses.transpose('presentation', 'neuroid')
        objectome = self.to_xarray(objectome)
        i2n = I2n()
        score = i2n(feature_responses, objectome)
        score = score.sel(aggregation='center')
        assert score == approx(expected_score, abs=0.01), f"expected {expected_score}, but got {score}"

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

    def to_xarray(self, objectome):
        columns = objectome.columns
        objectome = xr.DataArray(objectome['choice'],
                                 coords={column: ('presentation', objectome[column]) for column in columns},
                                 dims=['presentation'])
        objectome = objectome.rename({'id': 'image_id'})
        objectome['truth'] = objectome['label']
        objectome = objectome.set_index(presentation=[col if col != 'id' else 'image_id' for col in columns])
        return objectome


def get_all_features():
    for model, layer in [
        ('alexnet', 'classifier.5'),
        ('resnet-34', 'avgpool'),
        ('resnet-18', 'avgpool'),
        ('squeezenet1_0', 'features.12.expand3x3_activation'),
        ('squeezenet1_1', 'features.12.expand3x3_activation'),
    ]:
        print(model)
        get_features(model_name=model, layer=layer)


def get_features(model_name, layer):
    from candidate_models import base_models
    from glob import glob
    model = base_models.base_model_pool[model_name]
    stimuli_paths = list(glob('/braintree/home/msch/brain-score_packaging/objectome/objectome-224/*.png'))
    model(layers=[layer], stimuli_identifier='objectome-240', stimuli=stimuli_paths)
