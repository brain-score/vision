from pytest import approx

from brainscore.benchmarks.imagenet import Imagenet2012


class TestImagenet2012:
    def test_groundtruth(self):
        benchmark = Imagenet2012()
        source = benchmark._stimulus_set

        def candidate(stimuli):
            return source[source['image_id'] == stimuli['image_id']]['synset'].values

        score = benchmark(candidate)
        assert score.sel(aggregation='center') == approx(1)
