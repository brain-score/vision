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

        # import matplotlib.pyplot as plt
        # models = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
        #           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # # bs = [0, 0, 0, 0, 0, 0.060, .1599, .26, 0.36, 0.4599, 0.56,
        # #       0.66, 0.76, 0.86, 0.96, 0.94, 0.84, 0.74, 0.64, 0.54, 0.44]
        #
        # bs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.100, 0.20,
        #           0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.9, 0.8]
        # plt.scatter(models, bs)
        # plt.show()


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
        choice_labels = ["cube", "square", "y"]
        candidate.start_task(BrainModel.Task.label, choice_labels)

        def get_shapes(shape):
            shapes = self._assembly.stimulus_set[self._assembly.stimulus_set["image_shape"] == shape]
            shape_stimuli = place_on_screen(shapes, target_visual_degrees=candidate.visual_degrees(),
                                           source_visual_degrees=self._visual_degrees)
            return shape_stimuli

        def calculate_indexes():
            cube_stimuli = get_shapes(shape="cube")
            shape_stimuli = get_shapes(shape=self.shape)
            d_1 = candidate.response_time_proxy(stimuli=cube_stimuli)
            d_2 = candidate.response_time_proxy(stimuli=shape_stimuli)
            return (d_1 - d_2) / (d_1 + d_2)

        model_index = calculate_indexes()

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
