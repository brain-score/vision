from typing import Union

import numpy as np
import hashlib

from brainio.assemblies import NeuroidAssembly, DataAssembly
from brainscore_core import Score
from brainscore_vision.model_interface import BrainModel


class PrecomputedFeatures(BrainModel):
    def __init__(self, features: Union[DataAssembly, dict], visual_degrees):
        """
        :param features: The precomputed features. Either an assembly of features, indexable with `stimulus_id` or
            a dictionary mapping from stimulus identifier to feature assemblies.
        :param visual_degrees: Some visual degrees to use for the precomputed features. Since features are precomputed,
            this should only affect the `place_on_screen` in the benchmark's __call__ method.
        """
        self.features = features
        self._visual_degrees = visual_degrees

    @property
    def identifier(self) -> str:
        # serialize the features to a string and create hash
        features_data = str(self.features)
        features_hash = hashlib.md5(features_data.encode('utf-8')).hexdigest()
        return f"precomputed-{features_hash}"

    def visual_degrees(self) -> int:
        return self._visual_degrees

    def start_task(self, task, fitting_stimuli=None):
        pass

    def start_recording(self, region, *args, **kwargs):
        pass

    def look_at(self, stimuli, number_of_trials=1):
        features = self.features[stimuli.identifier] if isinstance(self.features, dict) else self.features
        missing_stimulus_ids = set(stimuli['stimulus_id'].values) - set(features['stimulus_id'].values)
        assert not missing_stimulus_ids, f"stored features do not contain stimulus_ids {missing_stimulus_ids}"
        image_indices = [np.where(features['stimulus_id'].values == image_id)[0][0]
                         for image_id in stimuli['stimulus_id'].values]
        features = features.isel(presentation=image_indices)
        assert all(features['stimulus_id'].values == stimuli['stimulus_id'].values)
        return features


def check_standard_format(assembly, nans_expected=False):
    assert isinstance(assembly, NeuroidAssembly)
    assert set(assembly.dims).issuperset({'presentation', 'neuroid'})
    assert hasattr(assembly, 'stimulus_id')
    assert hasattr(assembly, 'neuroid_id')
    if not nans_expected:
        assert not np.isnan(assembly).any()
    assert 'stimulus_set_identifier' in assembly.attrs


def bound_score(score: Score):
    """
    Force score value to be between 0 and 1.
    If score is lower than 0, set to 0. If score is greater than 1, set to 1.
    """
    if score < 0:
        score.loc[()] = 0
    elif score > 1:
        score.loc[()] = 1
