import numpy as np

from brainio_base.assemblies import NeuroidAssembly
from brainscore.model_interface import BrainModel


class PrecomputedFeatures(BrainModel):
    def __init__(self, features, visual_degrees):
        self.features = features
        self._visual_degrees = visual_degrees

    def visual_degrees(self) -> int:
        return self._visual_degrees

    def start_task(self, task, fitting_stimuli=None):
        pass

    def start_recording(self, region, *args, **kwargs):
        pass

    def look_at(self, stimuli, number_of_trials=1):
        missing_image_ids = set(stimuli['image_id'].values) - set(self.features[stimuli.identifier]['image_id'].values)
        assert not missing_image_ids, f"stored features do not contain image_ids {missing_image_ids}"
        features = self.features[stimuli.identifier].isel(
            presentation=[np.where(self.features[stimuli.identifier]['image_id'].values == image_id)[0][0]
                          for image_id in stimuli['image_id'].values])
        assert all(features['image_id'].values == stimuli['image_id'].values)
        return self.features[stimuli.identifier]


def check_standard_format(assembly):
    assert isinstance(assembly, NeuroidAssembly)
    assert set(assembly.dims).issuperset({'presentation', 'neuroid'})
    assert hasattr(assembly, 'image_id')
    assert hasattr(assembly, 'neuroid_id')
    assert not np.isnan(assembly).any()
    assert 'stimulus_set_identifier' in assembly.attrs
