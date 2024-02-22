import numpy as np
import pandas as pd
from brainio.stimuli import StimulusSet
from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.model_interface import BrainModel, BehavioralAssembly
from brainscore_vision.metrics import Score

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
        self._visual_degrees = 8
        self._number_of_trials = 1
        self._assembly = load_dataset('Hebart2023')
        self._stimulus_set = load_stimulus_set('Hebart2023')

        super().__init__(
            identifier=f'Hebart2023Accuracy_{similarity_measure}', version=1,
            ceiling_func=lambda: Score([0.6844, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation']),
            parent='Hebart2023',
            bibtex=BIBTEX
        )

    def __call__(self, candidate: BrainModel):
        # Create the new StimulusSet
        self.triplets = np.array([
            self._assembly.coords["image_1"].values,
            self._assembly.coords["image_2"].values,
            self._assembly.coords["image_3"].values
        ]).T.reshape(-1, 1)

        stimuli_data = [self._stimulus_set.loc[stim] for stim in self.triplets]
        stimuli = pd.concat(stimuli_data)
        stimuli.columns = self._stimulus_set.columns

        stimuli = StimulusSet(stimuli)
        stimuli.identifier = 'Hebart2023'
        stimuli.stimulus_paths = self._stimulus_set.stimulus_paths
        stimuli['stimulus_id'] = stimuli['stimulus_id'].astype(int)

        # Prepare the stimuli
        candidate.start_task(BrainModel.Task.odd_one_out)
        stimuli = place_on_screen(
            stimulus_set=stimuli,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees
        )

        # Run the model
        choices = candidate.look_at(stimuli, self._number_of_trials)

        # Score the model
        correct_choices = choices == self._assembly.coords["image_3"]
        raw_score = np.sum(correct_choices) / len(choices)
        score = (raw_score - 1 / 3) / (self.ceiling - 1 / 3)
        score[0] = max(0, score[0])
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = self.ceiling
        return score
