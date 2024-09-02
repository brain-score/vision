import numpy as np
import os
import json
import pandas as pd

from brainio.assemblies import BehavioralAssembly
from brainscore_core import Score
from brainscore_vision import load_metric, load_stimulus_set
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.model_interface import BrainModel


class SSV2ActivityRecognitionAccuracy(BenchmarkBase):
    def __init__(self):
        self._stimulus_set  = load_stimulus_set("SSV2ActivityRec2024")
        self._visual_degrees = 8
        self._similarity_metric = load_metric('accuracy')
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(SSV2ActivityRecognitionAccuracy, self).__init__(identifier='ssv2-accuracy', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='SSV2',
                                           bibtex="""@article{DBLP:journals/corr/GoyalKMMWKHFYMH17,
                                                      author       = {Raghav Goyal and
                                                                      Samira Ebrahimi Kahou and
                                                                      Vincent Michalski and
                                                                      Joanna Materzynska and
                                                                      Susanne Westphal and
                                                                      Heuna Kim and
                                                                      Valentin Haenel and
                                                                      Ingo Fr{\"{u}}nd and
                                                                      Peter Yianilos and
                                                                      Moritz Mueller{-}Freitag and
                                                                      Florian Hoppe and
                                                                      Christian Thurau and
                                                                      Ingo Bax and
                                                                      Roland Memisevic},
                                                      title        = {The "something something" video database for learning and evaluating
                                                                      visual common sense},
                                                      journal      = {CoRR},
                                                      year         = {2017},

                                                }""")

    

    def __call__(self, candidate):
        # prepare fitting stimuli
        fitting_stimuli = self._stimulus_set[self._stimulus_set['train'] == 1]
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        # prepare test stimuli
        test_stimuli = self._stimulus_set[self._stimulus_set['train'] == 0]
        test_stimuli = place_on_screen(test_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        
        candidate.start_task(BrainModel.Task.video_readout, fitting_stimuli)
        predictions = candidate.look_at(test_stimuli)
        score = self._similarity_metric(
            predictions['choice'],
            test_stimuli['label']
        )
        return score
