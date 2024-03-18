import numpy as np

from result_caching import store
from .registry import load_dataset, load_stimulus_set
from brainscore_vision.utils import LazyLoad
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.model_interface import BrainModel
from brainscore.metrics.ceiling import _SplitHalvesConsistency
from brainscore.metrics import Score
from brainio.assemblies import walk_coords, BehavioralAssembly


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
    acc_min = np.min([acc1, acc2], axis=0)
    intersection = np.trapz(acc_min, cohs)
    acc_max = np.max([acc1, acc2], axis=0)
    union = np.trapz(acc_max, cohs)
    score = intersection / (union+1e-8)
    return score

class CoherenceConsistency:
  def __call__(self, x, y):
      x = x.correct.groupby('coh').mean()
      y = y.correct.groupby('coh').mean()
      cohs = x.coh.values
      score = consistency_ding2012(cohs, x, y)
      score = Score(score)
      return score
consistency = CoherenceConsistency()
internal_consistency = _SplitHalvesConsistency(consistency, cross_validation_kwargs=dict(stratification_coord='coh'))


def metric(assembly, labelings):
    monkey = assembly.correct.groupby('coh').mean()
    model = (labelings==labelings.truth).groupby('coh').mean().sortby(monkey.cohs)
    cohs = monkey.cohs.values
    score = consistency_ding2012(cohs, monkey, model)
    score = Score(score)

def probabilities_to_label(probabilities):
    vals = probabilities.values
    pres_coords = {coord: (dims, values) for coord, dims, values in walk_coords(probabilities)
                             if set(dims) != {'choice'}}
    choices = probabilities.choice.values
    labels = np.argmax(vals, axis=1)
    labels = [[choices[i]] for i in labels]
    return BehavioralAssembly(labels, coords=pres_coords, dims=['presentation', 'choice'])


class Ding2012(BenchmarkBase):
    # behavioral benchmark
    def __init__(self):
        self._metric = metric
        self._assembly = LazyLoad(lambda: load_dataset("Ding2012"))
        self._visual_degrees = VISUAL_DEGREES
        self._number_of_trials = 1

        super().__init__(
            identifier=f'Ding2012', version=1,
            ceiling_func=lambda: internal_consistency(self._assembly),
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        fitting_stimuli = load_stimulus_set('Ding2012.train_stimuli')
        testing_stimuli = load_stimulus_set('Ding2012.test_stimuli')
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        probabilities = candidate.look_at(testing_stimuli, number_of_trials=self._number_of_trials)
        labels = probabilities_to_label(probabilities)
        raw_score = self._metric(self._assembly, labels)
        ceiling = self.ceiling
        score = raw_score / ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score
