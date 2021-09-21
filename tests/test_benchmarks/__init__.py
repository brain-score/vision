import numpy as np
from typing import Union

from brainio.assemblies import NeuroidAssembly, DataAssembly
from brainscore.model_interface import BrainModel


class PrecomputedFeatures(BrainModel):
    def __init__(self, features: Union[DataAssembly, dict], visual_degrees):
        """
        :param features: The precomputed features. Either an assembly of features, indexable with `image_id` or
            a dictionary mapping from stimulus identifier to feature assemblies.
        :param visual_degrees: Some visual degrees to use for the precomputed features. Since features are precomputed,
            this should only affect the `place_on_screen` in the benchmark's __call__ method.
        """
        self.features = features
        self._visual_degrees = visual_degrees

    def visual_degrees(self) -> int:
        return self._visual_degrees

    def start_task(self, task, fitting_stimuli=None):
        pass

    def start_recording(self, region, *args, **kwargs):
        pass

    def look_at(self, stimuli, number_of_trials=1):
        features = self.features[stimuli.identifier] if isinstance(self.features, dict) else self.features
        missing_image_ids = set(stimuli['image_id'].values) - set(features['image_id'].values)
        assert not missing_image_ids, f"stored features do not contain image_ids {missing_image_ids}"
        image_indices = [np.where(features['image_id'].values == image_id)[0][0]
                         for image_id in stimuli['image_id'].values]
        features = features.isel(presentation=image_indices)
        assert all(features['image_id'].values == stimuli['image_id'].values)
        return features


def check_standard_format(assembly):
    assert isinstance(assembly, NeuroidAssembly)
    assert set(assembly.dims).issuperset({'presentation', 'neuroid'})
    assert hasattr(assembly, 'image_id')
    assert hasattr(assembly, 'neuroid_id')
    assert not np.isnan(assembly).any()
    assert 'stimulus_set_identifier' in assembly.attrs
