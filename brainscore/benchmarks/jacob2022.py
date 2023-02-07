import numpy as np
import numpy.random

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
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

    def __init__(self):
        self._metric = Accuracy()
        self._assembly = LazyLoad(lambda: load_assembly('Jacob2020_3dpi'))

        # confirm VD
        self._visual_degrees = 8.0

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

        square_index = calculate_indexes("square")
        y_index = calculate_indexes("y")

        human_index = 0.76

        # per condition:
        model_score = max((1 - ((np.abs(human_index - y_index)) / human_index)), 0)


        raw_score = self._metric(labels, self._assembly)
        #     ceiling = self._ceiling(self._assembly)
        # score = raw_score / ceiling
        #
        # # cap score at 1 if ceiled score > 1
        # score[0] = 1 if score[0] > 1 else score[0]
        #
        # score.attrs['raw'] = raw_score
        # score.attrs['ceiling'] = ceiling
        # return score
        return 1


def Jacob20203dpi():
    return _Jacob20203DProcessingIndex()


def load_assembly(dataset):
    assembly = brainscore.get_assembly(dataset)
    return assembly


def load_stimulus_set(dataset):
    stimulus_set = brainscore.get_stimulus_set(dataset)
    return stimulus_set
#
#
# # ceiling method:
# class SplitHalvesConsistencyBaker:
#     def __init__(self, num_splits: int, split_coordinate: str, consistency_metric, image_types):
#         """
#         :param num_splits: how many times to create two halves
#         :param split_coordinate: over which coordinate to split the assembly into halves
#         :param consistency_metric: which metric to use to compute the consistency of two halves
#         """
#         self.num_splits = num_splits
#         self.split_coordinate = split_coordinate
#         self.consistency_metric = consistency_metric
#         self.image_types = image_types
#
#     def __call__(self, assembly) -> Score:
#
#         consistencies, uncorrected_consistencies = [], []
#         splits = range(self.num_splits)
#         random_state = np.random.RandomState(0)
#         for _ in splits:
#             num_subjects = len(set(assembly["subject"].values))
#             half1_subjects = random_state.choice(range(1, num_subjects), (num_subjects // 2), replace=False)
#             half1 = assembly[
#                 {'presentation': [subject in half1_subjects for subject in assembly['subject'].values]}]
#             half2 = assembly[
#                 {'presentation': [subject not in half1_subjects for subject in assembly['subject'].values]}]
#             consistency = self.consistency_metric(half1, half2)
#             uncorrected_consistencies.append(consistency)
#             # Spearman-Brown correction for sub-sampling
#             corrected_consistency = 2 * consistency / (1 + (2 - 1) * consistency)
#             consistencies.append(corrected_consistency)
#         consistencies = Score(consistencies, coords={'split': splits}, dims=['split'])
#         uncorrected_consistencies = Score(uncorrected_consistencies, coords={'split': splits}, dims=['split'])
#         average_consistency = consistencies.median('split')
#         average_consistency.attrs['raw'] = consistencies
#         average_consistency.attrs['uncorrected_consistencies'] = uncorrected_consistencies
#         return average_consistency