import numpy as np
import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.data_cloud_comparision import DataCloudComparison, get_means_stds, data_to_indexes
from brainscore.model_interface import BrainModel
from scipy.spatial import distance
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


class _Jacob20203DProcessingIndex(BenchmarkBase):
    """
    INFORMATION:

    Fundamentally, humans and models are performing different tasks in this benchmark. Models are passively viewing
    a stimuli (and having those activations retrieved from IT), whereas humans performed a visual search task
    in the papers listed below. The reasons for doing so are to replicate benchmark work done in Jacob et al 2020.

    Following Jacobs et al. 2020, we retrieve internal neural activity from the model (layer IT)
    rather than having the model perform the same visual search task that humans were performing
    in the behavioral experiment from which we take the behavioral data to compare against (below):

     1) Square Benchmark:
        - Enns and Rensink 1991
        - https://www2.psych.ubc.ca/~rensink/publications/download/E&R-PsychRev-91.pdf -> Page 4, Experiment 1
     2) Y Benchmark:
        - Enns and Rensink 1990
        - https://www.jstor.org/stable/40062736?seq=2 -> Page 324, Experiment 1

    Benchmark Choices:

    1) The Recording Target (IT) is an unfounded choice, but IT is generally considered a high-level representation
        that is only a linear step away from behavior (cf. Majaj*, Hong*, et al. 2015)
    2)  Time bins 70-170ms: unfounded choice. 70-170ms is generally considered the standard feedforward
        pass in the visual ventral stream.
    3) Number of Trials: 30- Unfounded choice because the data we test against is behavioral. 30 Seemed reasonable.

    NOTES:
    1) Jacob 2020 Reports a Square Human Baseline (from paper 1 above) of 0.76
        - Current Metric (Data Cloud Comparison) is generating an overage of 0.741, STD of 0.066
    2) Jacob 2020 reports a Y Human baseline (from paper 2 above) of 0.40
        - Current Metric (Data Cloud Comparison) is generating an average of 0.419, STD of 0.059

    """
    def __init__(self, discriminator):
        self._assembly = LazyLoad(lambda: load_assembly('Jacob2020_3dpi'))

        # confirm VD
        self._visual_degrees = 8

        self.discriminator = discriminator  # shape: y or square
        self.display_sizes = [1, 6, 12]  # used for slope calculations
        self._metric = DataCloudComparison()
        self._number_of_trials = 30

        super(_Jacob20203DProcessingIndex, self).__init__(
            identifier='Jacob2020_3dpi', version=1,
            ceiling_func=lambda: Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation']),
            parent='Jacob2020',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        candidate.start_recording(recording_target="IT", time_bins=[(70, 170)])
        stimuli = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                  source_visual_degrees=self._visual_degrees)
        it_recordings = candidate.look_at(stimuli, number_of_trials=self._number_of_trials)

        # grab activations per image (4 images total) from region IT
        cube_1_activations = it_recordings.sel(stimulus_id="cube_1")
        cube_2_activations = it_recordings.sel(stimulus_id="cube_2")
        shape_1_activations = it_recordings.sel(stimulus_id=f"{self.discriminator}_1")
        shape_2_activations = it_recordings.sel(stimulus_id=f"{self.discriminator}_2")

        # calculate distance between cubes, shapes.
        d1 = distance.euclidean(cube_1_activations, cube_2_activations)
        d2 = distance.euclidean(shape_1_activations, shape_2_activations)

        model_index = (d1 - d2) / (d1 + d2)

        # use cube 2 if shape is y. Cube 2's slope value is 12, which is what is needed
        if self.discriminator == "y":
            cube_means, cube_stds = get_means_stds("cube_2", self._assembly)
        else:
            cube_means, cube_stds = get_means_stds("cube_1", self._assembly)
        shape_means, shape_stds = get_means_stds(f"{self.discriminator}_1", self._assembly)
        human_indexes = data_to_indexes(self.display_sizes, cube_means, cube_stds, shape_means, shape_stds)
        human_indexes_average, human_indexes_error = np.mean(human_indexes), np.std(human_indexes)

        raw_score = self._metric(model_index, human_indexes)
        ceiling = self._metric.ceiling(human_indexes)
        score = raw_score / ceiling.sel(aggregation='center')

        # cap score at 1 if ceiled score > 1
        if score[(score['aggregation'] == 'center')] > 1:
            score.__setitem__({'aggregation': score['aggregation'] == 'center'}, 1)

        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


def Jacob20203dpi_square():
    return _Jacob20203DProcessingIndex(discriminator="square")


def Jacob20203dpi_y():
    return _Jacob20203DProcessingIndex(discriminator="y")


def load_assembly(dataset):
    assembly = brainscore.get_assembly(dataset)
    return assembly
