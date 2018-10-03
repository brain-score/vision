import os

import numpy as np
import pandas as pd
import xarray as xr
from pytest import approx

from brainscore.metrics.behavior import I2n
from brainscore.metrics.transformations import subset


class TestI2N(object):
    def test_objectome(self):
        packaged_filepath = os.path.join(os.path.dirname(__file__), 'objectome240.pkl')
        if not os.path.isfile(packaged_filepath):
            # repackage subsampled
            objectome = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'objectome24s100_humanpool.pkl'))
            subsample = pd.read_pickle(
                os.path.join(os.path.dirname(__file__), 'objectome24s100_imgsubsampled240_pandas.pkl'))
            objectome = objectome[objectome['id'].isin(subsample.values[:, 0])]
            objectome.to_pickle(packaged_filepath)
        else:
            objectome = pd.read_pickle(packaged_filepath)

        objectome['correct'] = objectome['choice'] == objectome['sample_obj']
        objectome['label'] = objectome['sample_obj']
        labels = {row['id']: row['label'] for _, row in objectome[['id', 'label']].drop_duplicates().iterrows()}
        # reformat to xarray
        columns = [column for column in objectome.columns]
        objectome = xr.DataArray(objectome['choice'],
                                 coords={column: ('presentation', objectome[column]) for column in columns},
                                 dims=['presentation'])
        objectome = objectome.rename({'id': 'image_id'})
        objectome['truth'] = objectome['label']
        objectome = objectome.set_index(presentation=[col if col != 'id' else 'image_id' for col in columns])

        feature_responses = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'resnet34_activations.pkl'))
        feature_responses = feature_responses['data']  # only layer is 'avgpool'
        image_ids = xr.DataArray(np.zeros(len(labels)), coords={'image_id': list(labels.keys())},
                                 dims=['image_id']).stack(presentation=['image_id'])
        feature_responses = subset(feature_responses, image_ids)
        labels = [labels[image_id] for image_id in feature_responses['image_id'].values]
        feature_responses['label'] = 'presentation', labels
        feature_responses = feature_responses.transpose('presentation', 'neuroid')

        i2n = I2n()
        score = i2n(feature_responses, objectome)
        score = i2n.aggregate(score)
        expected_score = .245
        assert score == approx(expected_score, abs=0.01), f"expected {expected_score}, but got {score}"
