import numpy as np
import numpy.random

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.data_cloud_comparision import DataCloudComparison
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad

BIBTEX = """@article{jacob2021qualitative,
              title={Qualitative similarities and differences in visual object representations between brains and deep networks},
              author={Jacob, Georgin and Pramod, RT and Katti, Harish and Arun, SP},
              journal={Nature communications},
              volume={12},
              number={1},
              pages={1872},
              year={2021},
              publisher={Nature Publishing Group UK London}
            }"""

DATASETS = ['3dpi']


class _Jacob20203DProcessingIndex(BenchmarkBase):

    def __init__(self, shape):
        self._assembly = LazyLoad(lambda: load_assembly('Jacob2020_3dpi'))

        # confirm VD
        self._visual_degrees = 8

        self.shape = shape
        self._metric = DataCloudComparison(shape=self.shape)
        self._number_of_trials = 1

        super(_Jacob20203DProcessingIndex, self).__init__(
            identifier='Jacob2020_3dpi', version=1,
            ceiling_func=lambda: Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation']),
            parent='Jacob2020',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):

        # calculates the maximum distance between two images by adding each of their distances from 0
        def calculate_distance(response_times, shape):
            image_1_distance = 0.3 * (response_times.sel(stimulus_id=f"{shape}_1").values[0][0]) - 900
            image_2_distance = 0.3 * (response_times.sel(stimulus_id=f"{shape}_2").values[0][0]) - 900
            distance = image_1_distance + image_2_distance
            return distance

        stimuli = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                        source_visual_degrees=self._visual_degrees)
        candidate.start_task(BrainModel.Task.response_time, stimuli)
        response_times = candidate.look_at(stimuli, number_of_trials=1)
        d1 = calculate_distance(response_times, "cube")
        d2 = calculate_distance(response_times, self.shape)
        model_index = (d1 - d2) / (d1 + d2)

        raw_score, ceiling = self._metric(model_index, self._assembly)

        score = raw_score / ceiling.sel(aggregation='center')
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling

        return raw_score


def Jacob20203dpi_square():
    return _Jacob20203DProcessingIndex(shape="square")


def Jacob20203dpi_y():
    return _Jacob20203DProcessingIndex(shape="y")


def load_assembly(dataset):
    assembly = brainscore.get_assembly(dataset)
    return assembly
