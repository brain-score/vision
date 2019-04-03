from pytest import approx

from brainscore.benchmarks.imagenet import Imagenet2012
from brainscore.model_interface import BrainModel


class TestImagenet2012:
    def test_groundtruth(self):
        benchmark = Imagenet2012()
        source = benchmark._stimulus_set

        class GroundTruth(BrainModel):
            def start_task(self, task):
                assert task == BrainModel.Task.probabilities

            def look_at(self, stimuli):
                return source[source['image_id'] == stimuli['image_id']]['synset'].values

        candidate = GroundTruth()
        score = benchmark(candidate)
        assert score.sel(aggregation='center') == approx(1)
