import numpy as np
import numpy.random

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.data_cloud_comparision import DataCloudComparison
from brainscore.model_interface import BrainModel
from brainscore.metrics import accuracy
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
        self._visual_degrees = 8.0

        self.shape = shape
        self._metric = DataCloudComparison(shape=self.shape)
        self._number_of_trials = 1

        super(_Jacob20203DProcessingIndex, self).__init__(
            identifier='Jacob2020_3dpi', version=1,
            ceiling_func=lambda: Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation']),
            parent='Jacob2020',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):

        candidate.start_task(BrainModel.Task.label, ["cube", "square", "y"])
        stimulus_set = load_stimulus_set("Jacob2020_3dpi")

        def get_shapes(shape):
            shapes = stimulus_set[stimulus_set["image_shape"] == shape]
            shape_stimuli = place_on_screen(shapes, target_visual_degrees=candidate.visual_degrees(),
                                           source_visual_degrees=self._visual_degrees)
            cube_1 = shape_stimuli[shape_stimuli["filename"] == f"{shape}_1.png"]
            cube_2 = shape_stimuli[shape_stimuli["filename"] == f"{shape}_2.png"]
            return cube_1, cube_2

        def calculate_indexes(shape):
            cube_1, cube_2 = get_shapes("cube")
            control_1, control_2 = get_shapes(shape)
            d_1 = candidate.response_time(image_1=cube_1, image_2=cube_2)
            d_2 = candidate.response_time(image_1=control_1, image_2=control_2)
            return (d_1 - d_2) / (d_1 + d_2)

        model_index = calculate_indexes(self.shape)

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


def load_stimulus_set(dataset):
    stimulus_set = brainscore.get_stimulus_set(dataset)
    return stimulus_set