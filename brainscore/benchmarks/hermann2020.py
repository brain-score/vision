from xarray import DataArray
import brainscore
from brainscore.model_interface import BrainModel
from brainscore.metrics.accuracy import Accuracy
from brainscore.benchmarks import BenchmarkBase, Score
import numpy as np


BIBTEX = """@article{hermann2020origins,
              title={The origins and prevalence of texture bias in convolutional neural networks},
              author={Hermann, Katherine and Chen, Ting and Kornblith, Simon},
              journal={Advances in Neural Information Processing Systems},
              volume={33},
              pages={19000--19015},
              year={2020}
        }""" 

        
class _Hermann2020Match(BenchmarkBase):
    def __init__(self,metric_identifier,stimulus_column):
        assert metric_identifier in ["shape_match","texture_match"]
        assert stimulus_column in ["original_image_category","conflict_image_category"]
        self._stimulus_set = brainscore.get_stimulus_set("brendel.Geirhos2021_cue-conflict")
        self._metric = Accuracy()
        self._number_of_trials = 1
        self._stimulus_column = stimulus_column
        super(_Hermann2020Match, self).__init__(
            identifier=f'brendel.Hermann2020-{metric_identifier}', version=1,
            ceiling_func=lambda: Score([1, np.nan],coords={'aggregation': ['center', 'error']}, dims=['aggregation']), 
            parent='brendel.Herman2020',
            bibtex=BIBTEX)
        
    def __call__(self,candidate: BrainModel):
        choice_labels = set(self._stimulus_set['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        labels = candidate.look_at(self._stimulus_set, number_of_trials=self._number_of_trials)
        #remove entries without cue-conflict
        labels = labels[0][labels["conflict_image_category"] != labels["original_image_category"]]
        target = self._stimulus_set[self._stimulus_set["conflict_image_category"] != self._stimulus_set["original_image_category"]]
        target = target[self._stimulus_column].values
        score = self._metric(labels,target)
        return score
    
    
def _shape_bias(candidate):
    shape_match = _Hermann2020Match("shape_match","original_image_category")(candidate)
    texture_match = _Hermann2020Match("texture_match","conflict_image_category")(candidate)
    return shape_match/(shape_match + texture_match)
    
    
Hermann2020cueconflictShapeMatch = lambda: _Hermann2020Match("shape_match","original_image_category")
Hermann2020cueconflictShapeBias = lambda: _shape_bias