from brainscore_core import Score

from brainscore_vision import load_stimulus_set, load_metric
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.model_interface import BrainModel

BIBTEX = """@article{hermann2020origins,
              title={The origins and prevalence of texture bias in convolutional neural networks},
              author={Hermann, Katherine and Chen, Ting and Kornblith, Simon},
              journal={Advances in Neural Information Processing Systems},
              volume={33},
              pages={19000--19015},
              year={2020},
              url={https://proceedings.neurips.cc/paper/2020/hash/db5f9f42a7157abe65bb145000b5871a-Abstract.html}
        }"""


class _Hermann2020Match(BenchmarkBase):
    def __init__(self, metric_identifier, stimulus_column):
        assert metric_identifier in ["shape_match", "texture_match"]
        assert stimulus_column in ["original_image_category", "conflict_image_category"]
        self._stimulus_set = load_stimulus_set("brendel.Geirhos2021_cue-conflict")
        self._metric = load_metric('accuracy')
        self._number_of_trials = 1
        self._stimulus_column = stimulus_column
        super(_Hermann2020Match, self).__init__(
            identifier=f'kornblith.Hermann2020-{metric_identifier}', version=1,
            ceiling_func=lambda: Score(1),
            parent='kornblith.Hermann2020',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._stimulus_set['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        labels = candidate.look_at(self._stimulus_set, number_of_trials=self._number_of_trials)
        # remove entries without cue-conflict
        labels = labels[0][labels["conflict_image_category"] != labels["original_image_category"]]
        cueconflict = self._stimulus_set["conflict_image_category"] != self._stimulus_set["original_image_category"]
        target = self._stimulus_set[cueconflict]
        target = target[self._stimulus_column].values
        score = self._metric(labels, target)
        return score


class Hermann2020cueconflictShapeBias(BenchmarkBase):
    def __init__(self):
        self.shape_benchmark = _Hermann2020Match("shape_match", "original_image_category")
        self.texture_benchmark = _Hermann2020Match("texture_match", "conflict_image_category")
        super(Hermann2020cueconflictShapeBias, self).__init__(
            identifier=f'brendel.Hermann2020-shape_bias', version=1,
            ceiling_func=lambda: Score(1),
            parent='brendel.Hermann2020',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        shape_match = self.shape_benchmark(candidate)
        texture_match = self.texture_benchmark(candidate)
        return shape_match / (shape_match + texture_match)


def Hermann2020cueconflictShapeMatch():
    return _Hermann2020Match("shape_match", "original_image_category")
