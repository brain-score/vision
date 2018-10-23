import os

import numpy as np
import pandas as pd
import xarray as xr
from pytest import approx

from brainscore.assemblies import DataAssembly
from brainscore.metrics.behavior import I2n
from brainscore.metrics.transformations import subset


class TestI2N(object):
    def test_match_rishi(self):
        objectome = self.get_objectome()
        objectome = objectome.sel(use=True)
        i2n = I2n()
        response_matrix = i2n.build_response_matrix_from_responses(objectome)
        response_matrix = i2n.normalize_response_matrix(response_matrix)

        expected = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'metrics240.pkl'))['I2_dprime_C']
        expected = expected[0][0]
        meta = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'meta.pkl'))
        sel = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'sel240.pkl'))
        meta = meta.loc[sel]
        choices = [meta['obj'].values[np.where(np.isnan(expected[:, choice_index]))[0][0]]
                   for choice_index in range(expected.shape[1])]
        presentation_coords = {column.replace('id', 'image_id'): ('presentation', meta[column].values)
                               for column in meta.columns}
        expected = DataAssembly(expected,
                                coords={**presentation_coords, **{'choice': choices}},
                                dims=['presentation', 'choice'])

        correlation = i2n.correlate(response_matrix, expected)
        assert correlation == approx(1)

    def test_resnet34(self):
        objectome = self.get_objectome()
        id_rows = {row['id']: row for _, row in objectome[['id', 'label', 'use']].drop_duplicates().iterrows()}

        feature_responses = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'resnet34_activations.pkl'))
        feature_responses = feature_responses['data']  # only layer is 'avgpool'
        image_ids = xr.DataArray(np.zeros(len(id_rows)), coords={'image_id': list(id_rows.keys())},
                                 dims=['image_id']).stack(presentation=['image_id'])
        feature_responses = subset(feature_responses, image_ids)
        labels = [id_rows[image_id]['label'] for image_id in feature_responses['image_id'].values]
        feature_responses['label'] = 'presentation', labels
        use = [id_rows[image_id]['use'] for image_id in feature_responses['image_id'].values]
        feature_responses['use'] = 'presentation', use
        feature_responses = feature_responses.transpose('presentation', 'neuroid')

        i2n = I2n()
        score = i2n(feature_responses, objectome)
        score = score.sel(aggregation='center')
        expected_score = .378
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
        # reformat to xarray
        columns = objectome.columns
        objectome = xr.DataArray(objectome['choice'],
                                 coords={column: ('presentation', objectome[column]) for column in columns},
                                 dims=['presentation'])
        objectome = objectome.rename({'id': 'image_id'})
        objectome['truth'] = objectome['label']
        objectome = objectome.set_index(presentation=[col if col != 'id' else 'image_id' for col in columns])
        return objectome
