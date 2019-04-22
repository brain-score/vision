from pytest import approx

from brainscore.benchmarks.imagenet import Imagenet2012
from brainscore.model_interface import BrainModel


class TestImagenet2012:
    def test_groundtruth(self):
        benchmark = Imagenet2012()
        source = benchmark._stimulus_set

        class GroundTruth(BrainModel):
            def start_task(self, task, fitting_stimuli):
                assert task == BrainModel.Task.label
                assert fitting_stimuli == 'imagenet'  # shortcut

            def look_at(self, stimuli):
                aligned_source = source[source['image_id'] == stimuli['image_id']]
                return aligned_source['synset'].values

        candidate = GroundTruth()
        score = benchmark(candidate)
        assert score.sel(aggregation='center') == approx(1)
