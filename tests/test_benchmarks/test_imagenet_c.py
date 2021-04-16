import numpy as np
import pytest
from pytest import approx

from brainio_base.assemblies import BehavioralAssembly
from brainscore.benchmarks.imagenet_c import Imagenet_C_Individual, Imagenet_C_Noise, Imagenet_C_Blur, \
    Imagenet_C_Weather, Imagenet_C_Digital
from brainscore.model_interface import BrainModel

# downloads all ImageNet C benchmarks (50.3G) and runs with default downsampling by a factor of 10
@pytest.mark.slow
class TestImagenetC:
    def test_groundtruth(self):
        benchmarks = [
            Imagenet_C_Noise(),
            Imagenet_C_Blur(),
            Imagenet_C_Weather(),
            Imagenet_C_Digital(),
        ]

        class Static(BrainModel):
            def start_task(self, task, fitting_stimuli):
                assert task == BrainModel.Task.label
                assert fitting_stimuli == 'imagenet'  # shortcut

            def look_at(self, stimuli, number_of_trials=1):
                labels = -np.ones_like(stimuli['image_id'].values)
                return BehavioralAssembly([labels], coords={
                    **{column: ('presentation', stimuli[column].values) for column in stimuli.columns},
                    **{'choice': ('choice', ['dummy'])}}, dims=['choice', 'presentation'])

        candidate = Static()
        scores = [benchmark(candidate) for benchmark in benchmarks]
        assert all([np.mean(score) == approx(0) for score in scores])

# downloads ImageNet C blur benchmarks (7.1G) and downsamples with a factor of 1000
class TestImagenetC_Category:
    def test_groundtruth(self):
        benchmarks = [
            Imagenet_C_Blur(factor=1000),
        ]

        class Static(BrainModel):
            def start_task(self, task, fitting_stimuli):
                assert task == BrainModel.Task.label
                assert fitting_stimuli == 'imagenet'  # shortcut

            def look_at(self, stimuli):
                labels = -np.ones_like(stimuli['image_id'].values)
                return BehavioralAssembly([labels], coords={
                    **{column: ('presentation', stimuli[column].values) for column in stimuli.columns},
                    **{'choice': ('choice', ['dummy'])}}, dims=['choice', 'presentation'])

        candidate = Static()
        scores = [benchmark(candidate) for benchmark in benchmarks]
        assert all([np.mean(score) == approx(0) for score in scores])
