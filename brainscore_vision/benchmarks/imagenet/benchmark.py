from pathlib import Path

import pandas as pd

from brainio.stimuli import StimulusSet, StimulusSetLoader
from brainscore_core import Score
from brainscore_vision import load_metric
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.model_interface import BrainModel

NUMBER_OF_TRIALS = 10


class Imagenet2012(BenchmarkBase):
    def __init__(self):
        # Not a valid StimulusSet if the files are not present on the filesystem.
        # We bypass StimulusSet.from_files() here.
        stimulus_set = pd.read_csv(Path(__file__).parent / 'imagenet2012.csv')
        StimulusSetLoader.correct_stimulus_id_name(stimulus_set)
        stimulus_set = StimulusSet(stimulus_set)
        stimulus_set.stimulus_paths = {row.stimulus_id: row.filepath for row in stimulus_set.itertuples()}
        self._stimulus_set = stimulus_set
        self._similarity_metric = load_metric('accuracy')
        ceiling = Score(1)
        super(Imagenet2012, self).__init__(identifier='ImageNet-top1', version=1,
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
