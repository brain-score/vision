from result_caching import store
from .registry import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, average_repetition
from brainscore_vision.utils import LazyLoad
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.model_interface import BrainModel
from brainscore.metrics.ceiling import _SplitHalvesConsistency
from brainscore.metrics import Score


VISUAL_DEGREES = 4
BIBTEX = """@article{ding2012neural,
  title={Neural correlates of perceptual decision making before, during, and after decision commitment in monkey frontal eye field},
  author={Ding, Long and Gold, Joshua I},
  journal={Cerebral cortex},
  volume={22},
  number={5},
  pages={1052--1067},
  year={2012},
  publisher={Oxford University Press}
}
"""

def consistency_ding2012(cohs, acc1, acc2):
    pass


class CoherenceConsistency:
  def __call__(self, x, y):
      x = x.correct.groupby('coh').mean()
      y = y.correct.groupby('coh').mean()
      cohs = x.coh.values
      score = consistency_ding2012(cohs, x, y)
      score = Score(score)
      return score
consistency = CoherenceConsistency()
internal_consistency = _SplitHalvesConsistency(consistency, stratification_coord='coh')


def metric(assembly, labelings):
    monkey = assembly.correct.groupby('coh').mean()
    model = labelings.groupby('coh').mean().sortby(monkey.cohs)
    cohs = monkey.cohs.values
    score = consistency_ding2012(cohs, monkey, model)
    score = Score(score)


class Ding2012(BenchmarkBase):
    # behavioral benchmark
    def __init__(self):
        self._metric = metric
        self._assembly = LazyLoad(lambda: load_assembly("Ding2012"))
        self._visual_degrees = VISUAL_DEGREES
        self._number_of_trials = 1

        super().__init__(
            identifier=f'Ding2012', version=1,
            ceiling_func=lambda: internal_consistency(self._assembly),
            parent='Geirhos2021',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        labels = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        raw_score = self._metric(labels, self._assembly)
        ceiling = self.ceiling
        score = raw_score / ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score
