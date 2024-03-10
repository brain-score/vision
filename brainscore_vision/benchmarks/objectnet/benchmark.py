import logging

from brainscore_core import Score
from brainscore_vision import load_stimulus_set, load_metric
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.model_interface import BrainModel

BIBTEX = """@inproceedings{DBLP:conf/nips/BarbuMALWGTK19,
                author    = {Andrei Barbu and
                            David Mayo and
                            Julian Alverio and
                            William Luo and
                            Christopher Wang and
                            Dan Gutfreund and
                            Josh Tenenbaum and
                            Boris Katz},
                title     = {ObjectNet: {A} large-scale bias-controlled dataset for pushing the
                            limits of object recognition models},
                booktitle = {NeurIPS 2019},
                pages     = {9448--9458},
                year      = {2019},
                url       = {https://proceedings.neurips.cc/paper/2019/hash/97af07a14cacba681feacf3012730892-Abstract.html},
                }"""

NUMBER_OF_TRIALS = 10

_logger = logging.getLogger(__name__)


class Objectnet(BenchmarkBase):
    def __init__(self):
        self._stimulus_set = load_stimulus_set('BarbuMayo2019')
        self._similarity_metric = load_metric('accuracy')
        ceiling = Score(1)

        super(Objectnet, self).__init__(identifier='BarbuMayo2019-top1', version=1,
                                        ceiling_func=lambda: ceiling,
                                        parent='engineering',
                                        bibtex=BIBTEX)

    def __call__(self, candidate):
        # The proper `fitting_stimuli` to pass to the candidate would be the imagenet training set.
        # For now, since almost all models in our hands were trained with imagenet, we'll just short-cut this
        # by telling the candidate to use its pre-trained imagenet weights.
        candidate.start_task(BrainModel.Task.label, 'imagenet')
        stimulus_set = self._stimulus_set[list(set(self._stimulus_set.columns) - {'synset'})]  # do not show label
        predictions = candidate.look_at(stimulus_set, number_of_trials=NUMBER_OF_TRIALS)
        score = self._similarity_metric(
            predictions.sortby('filename'),
            self._stimulus_set.sort_values('filename')['synset'].values
        )
        return score
