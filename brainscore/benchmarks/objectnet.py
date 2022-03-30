import os

import numpy as np
import pandas as pd

from brainio.stimuli import StimulusSet
from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
from brainscore.model_interface import BrainModel
from brainio.fetch import StimulusSetLoader

NUMBER_OF_TRIALS = 10

_logger = logging.getLogger(__name__)
LOCAL_STIMULUS_DIRECTORY = '/braintree/data2/active/common/objectnet-stimuli/'

class Objectnet(BenchmarkBase):
    def __init__(self):
        self.stimulus_set_name = f'katz.BarbuMayo2019.{noise_category}'
        self.stimulus_set = self.load_stimulus_set()

        self._similarity_metric = Accuracy()
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
       
        super(Objectnet, self).__init__(identifier='barbu_mayo2019-top1', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='ObjectNet',
                                           bibtex="""@inproceedings{DBLP:conf/nips/BarbuMALWGTK19,
                                                    author    = {Andrei Barbu and
                                                                David Mayo and
                                                                Julian Alverio and
                                                                William Luo and
                                                                Christopher Wang and
                                                                Dan Gutfreund and
                                                                Josh Tenenbaum and
                                                                Boris Katz},
                                                    title     = {ObjectNet: {A} large-scale bias-controlled dataset for pushing the
                                                                limits of object recognition models},
                                                    booktitle = {NeurIPS 2019},
                                                    pages     = {9448--9458},
                                                    year      = {2019},
                                                    url       = {https://proceedings.neurips.cc/paper/2019/hash/97af07a14cacba681feacf3012730892-Abstract.html},
                                                    }""")

    def load_stimulus_set(self):
        """
        ObjectNet is quite large, and thus cumbersome to download each time the benchmark is run.
        Here we try loading a local copy first, before proceeding to download the AWS copy.
        """
        try:
            _logger.debug(f'Loading local ObjectNet')
            loader = StimulusSetLoader(
                csv_path=os.path.join(
                            LOCAL_STIMULUS_DIRECTORY, 
                            f'katz.BarbuMayo2019.csv'
                        )
                stimuli_directory=LOCAL_STIMULUS_DIRECTORY
            )

            return loader.load()
        
        except OSError as error:
            _logger.debug(f'Excepted {error}. Attempting to access {self.stimulus_set_name} through Brainscore.')
            return brainscore.get_stimulus_set(self.stimulus_set_name)

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
