import numpy as np

import brainscore
from brainscore.benchmarks import BenchmarkBase
# TODO update imports vor brainscore_vision
from brainscore.benchmarks.screen import place_on_screen
from brainscore.model_interface import BrainModel
from brainscore.metrics import Score

BIBTEX = """@article{10.7554/eLife.82580,
          author = {Hebart, Martin N and Contier, Oliver and Teichmann, Lina and Rockter, Adam H and Zheng, Charles Y and Kidder, Alexis and Corriveau, Anna and Vaziri-Pashkam, Maryam and Baker, Chris I},
          journal = {eLife},
          month = {feb},
          pages = {e82580},
          title = {THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior},
          volume = 12,
          year = 2023
          }"""

class Hebart2023Accuracy(BenchmarkBase):
    def __init__(self, similarity_measure='dot'):
        self._similarity_measure = similarity_measure
        self._visual_degrees = 8
        self._number_of_trials = 1
        self._assembly = brainscore.get_assembly('Hebart2023')

        print(self._assembly.stimulus_set)

        super().__init__(
            identifier=f'Hebart2023Accuracy_{similarity_measure}', version=1,
            ceiling_func=lambda: Score([0.6844, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation']),
            parent='Hebart2023',
            bibtex=BIBTEX
        )

    def __call__(self, candidate: BrainModel):
        triplets = np.array([
            self._assembly.coords["image_1"].values,
            self._assembly.coords["image_2"].values,
            self._assembly.coords["image_3"].values
        ]).T

        # Do I look at the stimulus set or the assembly?
        fitting_stimuli = place_on_screen(
            stimulus_set=self._assembly.stimulus_set,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees
        )
        candidate.start_task(BrainModel.Task.odd_one_out, similarity_measure=self._similarity_measure)
        
        triplets = None
        
        choices = candidate.look_at(triplets, number_of_trials=self._number_of_trials)
        
        # This can probably stay as is
        correct_choices = choices == triplets[:, 2]
        raw_score = np.sum(correct_choices) / len(choices)
        score = (raw_score - 1 / 3) / (self.ceiling - 1 / 3)
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = self.ceiling
        return score
