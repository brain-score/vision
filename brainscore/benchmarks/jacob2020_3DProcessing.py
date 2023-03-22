import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics.data_cloud_comparision import DataCloudComparison
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

    """
    def __init__(self, shape):
        self._assembly = LazyLoad(lambda: load_assembly('Jacob2020_3dpi'))

        # confirm VD
        self._visual_degrees = 8

        self.shape = shape
        self._metric = DataCloudComparison(shape=self.shape)
        self._number_of_trials = 30

        super(_Jacob20203DProcessingIndex, self).__init__(
            identifier='Jacob2020_3dpi', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
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
        shape_1_activations = it_recordings.sel(stimulus_id=f"{self.shape}_1")
        shape_2_activations = it_recordings.sel(stimulus_id=f"{self.shape}_2")

        # calculate distance between cubes, shapes.
        d1 = distance.euclidean(cube_1_activations, cube_2_activations)
        d2 = distance.euclidean(shape_1_activations, shape_2_activations)

        model_index = (d1 - d2) / (d1 + d2)
        raw_score = self._metric(model_index, self._assembly)
        ceiling = self._ceiling(self._assembly)
        score = raw_score / ceiling.sel(aggregation='center')
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


def Jacob20203dpi_square():
    return _Jacob20203DProcessingIndex(shape="square")


def Jacob20203dpi_y():
    return _Jacob20203DProcessingIndex(shape="y")


def load_assembly(dataset):
    assembly = brainscore.get_assembly(dataset)
    return assembly
