import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics.data_cloud_comparision import DataCloudComparison
from brainscore.model_interface import BrainModel
from scipy.spatial import distance
from brainscore.metrics import Score
import numpy as np

from brainscore.utils import LazyLoad
from brainscore.metrics.data_cloud_comparision import DataCloudComparison, get_means_stds, data_to_indexes

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


class _Jacob2020OcclusionDepthOrdering(BenchmarkBase):
    """
    INFORMATION:

    Fundamentally, humans and models are performing different tasks in this benchmark. Models are passively viewing
    a stimuli (and having those activations retrieved from IT), whereas humans performed a visual search task
    in the papers listed below. The reasons for doing so are to replicate benchmark work done in Jacob et al 2020.

    Following Jacobs et al. 2020, we retrieve internal neural activity from the model (layer IT)
    rather than having the model perform the same visual search task that humans were performing
    in the behavioral experiment from which we take the behavioral data to compare against (below):

     1) Occlusion Benchmark:
        - Enns and Rensink 1998
        - https://www.sciencedirect.com/science/article/pii/S0042698998000510#FIG2 -> Figure 1, Conditions 1A, 1B, 1C
     2) Depth Ordering Benchmark:
        - Enns and Rensink 1998
        - https://www.sciencedirect.com/science/article/pii/S0042698998000510#FIG2 -> Figure 2, Conditions 2A, 2B, 2C

    Benchmark Choices:

    1) The Recording Target (IT) is an unfounded choice, but IT is generally considered a high-level representation
        that is only a linear step away from behavior (cf. Majaj*, Hong*, et al. 2015)
    2)  Time bins 70-170ms: unfounded choice. 70-170ms is generally considered the standard feedforward
        pass in the visual ventral stream.
    3) Number of Trials: 30- Unfounded choice because the data we test against is behavioral. 30 Seemed reasonable.

    """
    def __init__(self, discriminator):
        self._assembly = LazyLoad(lambda: load_assembly('Jacob2020_occlusion_depth_ordering'))
        self.display_sizes = [2, 8, 14]

        # confirm VD
        self._visual_degrees = 8

        self.discriminator = discriminator
        self._metric = DataCloudComparison()
        self._number_of_trials = 30

        super(_Jacob2020OcclusionDepthOrdering, self).__init__(
            identifier='Jacob2020_occlusion_depth_ordering', version=1,
            ceiling_func=lambda: Score([1, np.nan], coords={'aggregation': ['center', 'error']},
                                       dims=['aggregation']),
            parent='Jacob2020',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):

        # tell the model to start recording:
        # TODO

        # place the stimuli on screen/convert to visual degrees:
        # TODO

        # have the model look at the stimuli and generate the IT recordings
        # TODO

        # grab activations per image (3 images total) from region IT
        # TODO

        # calculate euclidean distance between images.
        # TODO

        # calculate model index, based on d1, d2
        # TODO

        # process assembly to get correct indexes:
        # occluded_means, occluded_stds = get_means_stds(f"{self.discriminator}_occluded", self._assembly)
        # notched_means, notched_stds = get_means_stds(f"{self.discriminator}_notched", self._assembly)
        #
        # human_indexes = data_to_indexes(self.display_sizes, occluded_means, occluded_stds, notched_means, notched_stds)

        # score the benchmark on the metric:
        #  raw_score = self._metric(model_index, human_indexes)

        # Calculate the ceiling:
        # TODO

        # divide raw score by ceiling to get score:
        # TODO

        # make sure score's attrs are set and return score:
        # TODO

        return 1


def Jacob2020_Occlusion():
    return _Jacob2020OcclusionDepthOrdering(discriminator="occlusion")


def Jacob2020_Depth_Ordering():
    return _Jacob2020OcclusionDepthOrdering(discriminator="depth_ordering")


def load_assembly(dataset):
    assembly = brainscore.get_assembly(dataset)
    return assembly
