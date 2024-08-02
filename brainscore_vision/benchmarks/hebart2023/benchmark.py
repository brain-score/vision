import numpy as np

from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers import bound_score
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics import Score
from brainscore_vision.model_interface import BrainModel

BIBTEX = """@article{10.7554/eLife.82580,
          author = {Hebart, Martin N and Contier, Oliver and Teichmann, Lina and Rockter, Adam H and Zheng, Charles Y and Kidder, Alexis and Corriveau, Anna and Vaziri-Pashkam, Maryam and Baker, Chris I},
          journal = {eLife},
          month = {feb},
          pages = {e82580},
          title = {THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior},
          volume = 12,
          year = 2023
          }"""
VISUAL_DEGREES = 8


class Hebart2023Match(BenchmarkBase):
    def __init__(self):
        self._visual_degrees = VISUAL_DEGREES
        self._number_of_trials = 1
        self._assembly = load_dataset('Hebart2023')
        self._stimulus_set = load_stimulus_set('Hebart2023')

        # The noise ceiling was computed by averaging the percentage of participants 
        # who made the same choice for a given triplet. See the paper for more detail.
        super().__init__(
            identifier='Hebart2023-match', version=1,
            ceiling_func=lambda: Score(0.6767),
            parent='behavior_vision',
            bibtex=BIBTEX
        )

    def set_number_of_triplets(self, n: int):
        """ Allows to reduce the number of triplets to reduce the compute requirements for debugging """
        self._assembly = self._assembly[:n]

    def __call__(self, candidate: BrainModel):
        # Create stimuli with triplets (all 3 consecutive stimuli form one trial following model_interface)
        self.triplets = np.array([
            self._assembly.coords["image_1"].values,
            self._assembly.coords["image_2"].values,
            self._assembly.coords["image_3"].values
        ]).T.reshape(-1)  # flatten into a list of stimuli ids

        triplet_stimuli = self._stimulus_set.loc[self.triplets]

        # Prepare the stimuli
        candidate.start_task(BrainModel.Task.odd_one_out)
        triplet_stimuli = place_on_screen(
            stimulus_set=triplet_stimuli,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees
        )

        # Run the model
        choices = candidate.look_at(triplet_stimuli, self._number_of_trials)

        # Score the model
        # We chose not to compute error estimates but you could compute them
        # by spliting the data into five folds and computing the standard deviation.
        correct_choices = choices.values == self._assembly.coords["image_3"].values  # third image is always correct
        raw_score = np.sum(correct_choices) / len(choices['presentation'])
        score = (raw_score - 1 / 3) / (self.ceiling - 1 / 3)
        bound_score(score)
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = self.ceiling
        return score
