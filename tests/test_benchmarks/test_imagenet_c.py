import numpy as np
import pytest
from pytest import approx

from brainio_base.assemblies import BehavioralAssembly
from brainscore.benchmarks.imagenet import Imagenet2012
from brainscore.benchmarks.imagenet_c import Imagenet_C_Individual, Imagenet_C_Noise, Imagenet_C_Blur, Imagenet_C_Weather, Imagenet_C_Digital
from brainscore.model_interface import BrainModel

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

            def look_at(self, stimuli):
                labels = -np.ones_like(stimuli['image_id'].values)
                return BehavioralAssembly([labels], coords={
                    **{column: ('presentation', stimuli[column].values) for column in stimuli.columns},
                    **{'choice': ('choice', ['dummy'])}}, dims=['choice', 'presentation'])

        candidate = Static()
        scores = [benchmark(candidate) for benchmark in benchmarks]
        assert all([np.mean(score) == approx(0) for score in scores])

class TestImagenetCIndividual:
    def test_groundtruth(self, name='dietterich.Hendrycks2019.gaussian_noise_1', noise_type='gaussian_noise'):
        benchmark = Imagenet_C_Individual(name, noise_type)
        source = benchmark._stimulus_set

        class GroundTruth(BrainModel):
            def start_task(self, task, fitting_stimuli):
                assert task == BrainModel.Task.label
                assert fitting_stimuli == 'imagenet'  # shortcut

            def look_at(self, stimuli):
                source_image_ids = source['image_id'].values
                stimuli_image_ids = stimuli['image_id'].values
                sorted_x = source_image_ids[np.argsort(source_image_ids)]
                sorted_index = np.searchsorted(sorted_x, stimuli_image_ids)
                aligned_source = source.loc[sorted_index]
                labels = aligned_source['synset'].values
                return BehavioralAssembly([labels], coords={
                    **{column: ('presentation', aligned_source[column].values) for column in aligned_source.columns},
                    **{'choice': ('choice', ['dummy'])}}, dims=['choice', 'presentation'])

        candidate = GroundTruth()
        assert np.mean(benchmark(candidate)) == approx(1)
