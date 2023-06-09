from typing import Tuple
import numpy as np
import xarray as xr

import brainscore
from brainio.assemblies import PropertyAssembly
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics.threshold import ThresholdElevation
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad


BIBTEX = """@article{malania2007,
            author = {Malania, Maka and Herzog, Michael H. and Westheimer, Gerald},
            title = "{Grouping of contextual elements that affect vernier thresholds}",
            journal = {Journal of Vision},
            volume = {7},
            number = {2},
            pages = {1-1},
            year = {2007},
            issn = {1534-7362},
            doi = {10.1167/7.2.1},
            url = {https://doi.org/10.1167/7.2.1}
        }"""

BASELINE_CONDITION = 'vernier-only'
DATASETS = ['short-2', 'short-4', 'short-6', 'short-8', 'short-16', 'equal-2', 'long-2', 'equal-16', 'long-16']
# Values in NUM_FLANKERS_PER_CONDITION denote the condition (i.e., in this case the number of flankers) to be selected
# This is kept track of simply because the benchmark uses threshold elevation - i.e., a comparison of 2 conditions
NUM_FLANKERS_PER_CONDITION = {'short-2': 2, 'short-4': 4, 'short-6': 6, 'short-8': 8,
                              'short-16': 16, 'equal-2': 2, 'long-2': 2, 'equal-16': 16,
                              'long-16': 16, 'vernier-only': 0}

for dataset in DATASETS:
    # behavioral benchmark
    identifier = f"Malania_{dataset.replace('-', '')}"
    globals()[identifier] = lambda dataset=dataset: _Malania2007Base(dataset)


class _Malania2007Base(BenchmarkBase):
    """
    INFORMATION:

    Benchmark DATASETS should be considered as independent. This means that participant-specific across-condition data
    should only ever be compared using the 'subject_unique_id'. In some conditions (short-2, vernier_only, short-16)
    an additional observer was added from the original paper's plots. This is because in these conditions, two
    experiments were independently conducted, and 1 additional observer that was non-overlapping between the
    experiments was added to the aggregate benchmark.

    While humans and models are performing the same testing task in this benchmark, there are a number of choices
    that are made in this benchmark that make minor deviations from the human experiment. The choices that make
    deviations from the human experiment are listed below alongside the reason for why the departure was made,
    and what the 'precisely faithful' alternative would be.

    Benchmark Choices:

    1) The number and type of fitting stimuli are unfounded choices. Currently, the number of fitting stimuli is chosen
        to be relatively small, but sufficient for good decoding performance in the baseline condition in general.
        - Precisely faithful alternative: Present text instructions to models as they were presented to humans
            * Why not this alternative? Since the experiment is about early visual perception, and there are currently
            few/no models capable of a task like this, it would not be interesting.
        - Somewhat faithful alternative: Present a smaller number of training stimuli, motivated by work like
        Lee & DiCarlo (2023), biorXiv (doi:https://doi.org/10.1101/2022.12.31.522402).
            * Why not this alternative? Since the experiment is not about perceptual learning but about early visual
            perception, and there are few/no models capable of a task like this, it would not be interesting.
    2) In the human experiment, stimuli were presented at exactly the foveal position. In the model experiment,
        testing stimuli are presented at exactly the foveal position +- 72arcsec = 0.02deg.
        * Why this alternative? Since most models evaluated are test-time deterministic, we want a more precise
        estimate of the threshold than a point estimate. Since human microsaccades of small distances are generally
        uncontrolled and uncontrollable for (e.g., up to 360arcsec = 6arcmin = 0.1 deg), we believe the tiny jitter
        of 0.02deg to have no impact at all on the comparison under study, while improving the precision of threshold
        estimates.

    """
    def __init__(self, condition: str):
        self.baseline_condition = BASELINE_CONDITION
        self.condition = condition

        # since this benchmark compares threshold elevation against a baseline, we omit one subject
        # in some conditions in which that subject did not perform both the baseline and the test
        # condition
        baseline_assembly = LazyLoad(lambda: load_assembly(self.baseline_condition))
        condition_assembly = LazyLoad(lambda: load_assembly(self.condition))
        self._assembly, self._baseline_assembly = remove_subjects_with_nans(condition_assembly,
                                                                            baseline_assembly)

        self._assemblies = {'baseline_assembly': self._baseline_assembly,
                            'condition_assembly': self._assembly}
        self._stimulus_set = brainscore.get_stimulus_set(f'{self.condition}')
        self._baseline_stimulus_set = brainscore.get_stimulus_set(f'{self.baseline_condition}')
        self._stimulus_sets = {self.condition: self._stimulus_set,
                               self.baseline_condition: self._baseline_stimulus_set}
        self._fitting_stimuli = brainscore.get_stimulus_set(f'{self.condition}_fit')

        self._metric = ThresholdElevation(independent_variable='vernier_offset',
                                          baseline_condition=self.baseline_condition,
                                          test_condition=self.condition,
                                          threshold_accuracy=0.75)
        self._ceiling = self._metric.ceiling(self._assemblies)

        self._visual_degrees = 2.986667
        self._number_of_trials = 1

        super(_Malania2007Base, self).__init__(
            identifier=f'Malania2007_{condition}', version=1,
            ceiling_func=lambda: self._ceiling,
            parent='Malania2007',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        model_responses = {}
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli=self._fitting_stimuli)
        for condition in (self.baseline_condition, self.condition):
            stimulus_set = place_on_screen(
                self._stimulus_sets[condition],
                target_visual_degrees=candidate.visual_degrees(),
                source_visual_degrees=self._visual_degrees
            )
            model_responses[condition] = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)

        raw_score = self._metric(model_responses, self._assemblies)

        # Adjust score to ceiling
        ceiling = self._ceiling
        score = raw_score / ceiling.sel(aggregation='center')

        # cap score at 1 if ceiled score > 1
        if score[(score['aggregation'] == 'center')] > 1:
            score.__setitem__({'aggregation': score['aggregation'] == 'center'}, 1)

        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


def load_assembly(dataset: str) -> PropertyAssembly:
    assembly = brainscore.get_assembly(f'Malania2007_{dataset}')
    return assembly


def remove_subjects_with_nans(condition_assembly: PropertyAssembly,
                              baseline_assembly: PropertyAssembly
                              ) -> Tuple[PropertyAssembly, PropertyAssembly]:
    # Find the indices of the subjects with NaN values in the first PropertyAssembly
    nan_subjects = np.isnan(condition_assembly.values)

    # Convert the boolean array to a DataArray with the same coordinates as the input assemblies
    nan_subjects_da = xr.DataArray(nan_subjects, coords=condition_assembly.coords, dims=condition_assembly.dims)

    # Filter out the subjects with NaN values from both PropertyAssemblies
    filtered_condition_assembly = condition_assembly.where(~nan_subjects_da, drop=True)
    filtered_baseline_assembly = baseline_assembly.where(~nan_subjects_da, drop=True)

    return filtered_condition_assembly, filtered_baseline_assembly
