import os
import pickle

import numpy as np

from brainio_base.assemblies import NeuroidAssembly
from brainscore.model_interface import BrainModel


class PrecomputedFeatures(BrainModel):
    def __init__(self, features):
        self.features = features

    def start_recording(self, region, *args, **kwargs):
        pass

    def look_at(self, stimuli):
        assert set(self.features['image_id'].values) == set(stimuli['image_id'].values)
        features = self.features.isel(presentation=[np.where(self.features['image_id'].values == image_id)[0][0]
                                                    for image_id in stimuli['image_id'].values])
        assert all(features['image_id'].values == stimuli['image_id'].values)
        return self.features


class StoredPrecomputedFeatures(PrecomputedFeatures):
    def __init__(self, filename):
        self.filepath = os.path.join(os.path.dirname(__file__), filename)
        assert os.path.isfile(self.filepath)
        with open(self.filepath, 'rb') as f:
            features = pickle.load(f)['data']
        super(StoredPrecomputedFeatures, self).__init__(features)


def check_standard_format(assembly):
    assert isinstance(assembly, NeuroidAssembly)
    assert set(assembly.dims).issuperset({'presentation', 'neuroid'})
    assert hasattr(assembly, 'image_id')
    assert hasattr(assembly, 'neuroid_id')
    assert not np.isnan(assembly).any()
    assert 'stimulus_set_name' in assembly.attrs
