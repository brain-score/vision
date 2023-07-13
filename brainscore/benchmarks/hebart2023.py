from brainscore.benchmarks import BenchmarkBase
from brainscore.model_interface import BrainModel


BIBTEX = """@article{10.7554/eLife.82580,
          author = {Hebart, Martin N and Contier, Oliver and Teichmann, Lina and Rockter, Adam H and Zheng, Charles Y and Kidder, Alexis and Corriveau, Anna and Vaziri-Pashkam, Maryam and Baker, Chris I},
          journal = {eLife},
          month = {feb},
          pages = {e82580},
          title = {THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior},
          volume = 12,
          year = 2023
          }"""

class _Hebart2023Accuracy(BenchmarkBase):
    # behavioral benchmark
    def __init__(self, dataset):
        self._metric = None
        self._assembly = None
        self._visual_degrees = None

        self._number_of_trials = None

        super(_Hebart2023Accuracy, self).__init__(
            identifier=None, version=1,
            ceiling_func=None,
            parent=None,
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        candidate.start_task(BrainModel.Task.odd_one_out, None)
        labels = candidate.look_at(None, number_of_trials=self._number_of_trials)
        raw_score = None
        ceiling = self.ceiling
        score = (raw_score - 1/3) / (ceiling - 1/3)
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score

