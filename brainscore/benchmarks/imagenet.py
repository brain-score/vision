import os

import numpy as np
import pandas as pd

from brainio.stimuli import StimulusSet
from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
from brainscore.model_interface import BrainModel

NUMBER_OF_TRIALS = 10


class Imagenet2012(BenchmarkBase):
    def __init__(self):
        stimulus_set = pd.read_csv(os.path.join(os.path.dirname(__file__), 'imagenet2012.csv'))
        stimulus_set = StimulusSet(stimulus_set)
        stimulus_set.image_paths = {row.stimulus_id: row.filepath for row in stimulus_set.itertuples()}
        self._stimulus_set = stimulus_set
        self._similarity_metric = Accuracy()
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(Imagenet2012, self).__init__(identifier='fei-fei.Deng2009-top1', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='ImageNet',
                                           bibtex="""@INPROCEEDINGS{5206848,  
                                                author={J. {Deng} and W. {Dong} and R. {Socher} and L. {Li} and  {Kai Li} and  {Li Fei-Fei}},  
                                                booktitle={2009 IEEE Conference on Computer Vision and Pattern Recognition},   
                                                title={ImageNet: A large-scale hierarchical image database},   
                                                year={2009},  
                                                volume={},  
                                                number={},  
                                                pages={248-255},
                                                url = {https://ieeexplore.ieee.org/document/5206848}
                                            }""")

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
