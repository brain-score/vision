import os
import pickle

import numpy as np

from brainscore.model_interface import BrainModel


class PrecomputedFeatures(BrainModel):
    def __init__(self, features, visual_degrees):
        self.features = features
        self._visual_degrees = visual_degrees

    def visual_degrees(self) -> int:
        return self._visual_degrees

    def start_recording(self, region, *args, **kwargs):
        pass

    def look_at(self, stimuli):
        assert set(self.features['image_id'].values) == set(stimuli['image_id'].values)
        features = self.features.isel(presentation=[np.where(self.features['image_id'].values == image_id)[0][0]
                                                    for image_id in stimuli['image_id'].values])
        assert all(features['image_id'].values == stimuli['image_id'].values)
        return self.features


class StoredPrecomputedFeatures(PrecomputedFeatures):
    def __init__(self, filename, visual_degrees):
        self.filepath = os.path.join(os.path.dirname(__file__), filename)
        assert os.path.isfile(self.filepath)
        with open(self.filepath, 'rb') as f:
            features = pickle.load(f)['data']
        super(StoredPrecomputedFeatures, self).__init__(features, visual_degrees=visual_degrees)
