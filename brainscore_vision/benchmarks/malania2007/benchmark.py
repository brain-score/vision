from typing import Tuple
import numpy as np
import xarray as xr

import brainscore_vision
from brainio.assemblies import PropertyAssembly
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision import load_dataset, load_stimulus_set, load_metric
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import LazyLoad
from brainscore_core.metrics import Score


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
DATASETS = ['short-2', 'short-4', 'short-6', 'short-8', 'short-16', 'equal-2', 'long-2', 'equal-16', 'long-16',
            'vernieracuity']
# Values in NUM_FLANKERS_PER_CONDITION denote the condition (i.e., in this case the number of flankers) to be selected
# This is kept track of simply because the benchmark uses threshold elevation - i.e., a comparison of 2 conditions
NUM_FLANKERS_PER_CONDITION = {'short-2': 2, 'short-4': 4, 'short-6': 6, 'short-8': 8,
                              'short-16': 16, 'equal-2': 2, 'long-2': 2, 'equal-16': 16,
                              'long-16': 16, 'vernier-only': 0}


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
        to be relatively large, and hopefully sufficient for decoding in the baseline condition in general.
        - Precisely faithful alternative: Present text instructions to models as they were presented to humans
            * Why not this alternative? Since the experiment is about early visual perception, and there are currently
            few/no models capable of a task like this, it would not be interesting.
        - Somewhat faithful alternative: Present a smaller number of training stimuli, motivated by work like
        Lee & DiCarlo (2023), biorXiv (doi:https://doi.org/10.1101/2022.12.31.522402).
            * Why not this alternative? Since the experiment is not about perceptual learning but about early visual
            perception, and there are few/no models capable of a task like this, it would not be interesting.
        - Importantly, this means the benchmark examines the models' capability to support a task like this, rather than
        their capability to learn a task like this.
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
        self._assembly, self._baseline_assembly = filter_baseline_subjects(condition_assembly,
                                                                           baseline_assembly)

        self._assemblies = {'baseline_assembly': self._baseline_assembly,
                            'condition_assembly': self._assembly}
        self._stimulus_set = brainscore_vision.load_stimulus_set(f'Malania2007_{self.condition}')
        self._baseline_stimulus_set = brainscore_vision.load_stimulus_set(f'Malania2007_{self.baseline_condition}')
        self._stimulus_sets = {self.condition: self._stimulus_set,
                               self.baseline_condition: self._baseline_stimulus_set}
        self._fitting_stimuli = brainscore_vision.load_stimulus_set(f'Malania2007_{self.condition}_fit')

        self._metric = load_metric('threshold_elevation',
                                   independent_variable='image_label',
                                   baseline_condition=self.baseline_condition,
                                   test_condition=self.condition,
                                   threshold_accuracy=0.75)

        self._visual_degrees = 2.986667
        self._number_of_trials = 10  # arbitrary choice for microsaccades to improve precision of estimates

        super(_Malania2007Base, self).__init__(
            identifier=f'Malania2007_{condition}', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assemblies),
            parent='Malania2007',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        model_responses = {}
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli=self._fitting_stimuli,
                             number_of_trials=self._number_of_trials, require_variance=True)
        for condition in (self.baseline_condition, self.condition):
            stimulus_set = place_on_screen(
                self._stimulus_sets[condition],
                target_visual_degrees=candidate.visual_degrees(),
                source_visual_degrees=self._visual_degrees
            )
            model_responses[condition] = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials,
                                                           require_variance=True)

        raw_score = self._metric(model_responses, self._assemblies)

        # Adjust score to ceiling
        ceiling = self.ceiling
        score = raw_score / ceiling.sel(aggregation='center')

        # cap score at 1 if ceiled score > 1
        if score[(score['aggregation'] == 'center')] > 1:
            score.__setitem__({'aggregation': score['aggregation'] == 'center'}, 1)

        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


class _Malania2007VernierAcuity(BenchmarkBase):
    def __init__(self):
        self.baseline_condition = BASELINE_CONDITION
        self.conditions = DATASETS.copy()
        self.conditions.remove('vernieracuity')

        self._assemblies = {condition: {'baseline_assembly': self.get_assemblies(condition)['baseline_assembly'],
                                        'condition_assembly': self.get_assemblies(condition)['condition_assembly']}
                            for condition in self.conditions}
        self._stimulus_set = brainscore_vision.load_stimulus_set(f'Malania2007_{self.baseline_condition}')
        self._fitting_stimuli = {condition: brainscore_vision.load_stimulus_set(f'Malania2007_{condition}_fit')
                               for condition in self.conditions}

        self._metric = load_metric('threshold',
                                   independent_variable='image_label',
                                   threshold_accuracy=0.75)

        self._visual_degrees = 2.986667
        self._number_of_trials = 10  # arbitrary choice for microsaccades to improve precision of estimates

        super(_Malania2007VernierAcuity, self).__init__(
            identifier=f'Malania2007_vernieracuity', version=1,
            ceiling_func=lambda: self.mean_ceiling(),
            parent='Malania2007',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        scores = []
        for condition in self.conditions:
            candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli=self._fitting_stimuli[condition],
                                 number_of_trials=self._number_of_trials, require_variance=True)
            stimulus_set = place_on_screen(
                self._stimulus_set,
                target_visual_degrees=candidate.visual_degrees(),
                source_visual_degrees=self._visual_degrees
            )
            model_response = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials,
                                               require_variance=True)

            raw_score = self._metric(model_response, self._assemblies[condition])
            # Adjust score to ceiling
            ceiling = self.ceiling
            score = raw_score / ceiling.sel(aggregation='center')

            # cap score at 1 if ceiled score > 1
            if score[(score['aggregation'] == 'center')] > 1:
                score.__setitem__({'aggregation': score['aggregation'] == 'center'}, 1)

            score.attrs['raw'] = raw_score
            score.attrs['ceiling'] = ceiling
            scores.append(score)
        # average all scores to get 1 average score
        mean_score = Score(np.mean(scores))
        mean_score.attrs['error'] = np.mean([score['error'] for score in scores])
        return mean_score

    def get_assemblies(self, condition: str):
        baseline_assembly = LazyLoad(lambda: load_assembly(self.baseline_condition))
        condition_assembly = LazyLoad(lambda: load_assembly(condition))
        assembly, baseline_assembly = filter_baseline_subjects(condition_assembly,
                                                               baseline_assembly)
        return {'condition_assembly': assembly,
                'baseline_assembly': baseline_assembly}

    def mean_ceiling(self):
        ceilings = []
        errors = []
        for assembly_name in self._assemblies.keys():
            this_ceiling = self._metric.ceiling(self._assemblies[assembly_name]['baseline_assembly'])
            ceilings.append(this_ceiling.values)
            errors.append(this_ceiling.error)
        mean_ceiling = Score(np.mean(ceilings))
        mean_ceiling.attrs['error'] = np.mean(errors)
        return mean_ceiling



def load_assembly(dataset: str) -> PropertyAssembly:
    assembly = brainscore_vision.load_dataset(f'Malania2007_{dataset}')
    return assembly


def filter_baseline_subjects(condition_assembly: PropertyAssembly,
                             baseline_assembly: PropertyAssembly
                             ) -> Tuple[PropertyAssembly, PropertyAssembly]:
    """A function to select only the unique subjects that exist in the condition_assembly."""
    non_nan_mask = ~np.isnan(condition_assembly.values)
    unique_ids = condition_assembly.coords['subject'][non_nan_mask].values.tolist()

    mask = baseline_assembly.coords['subject'].isin(unique_ids)
    filtered_baseline_assembly = baseline_assembly.where(mask, drop=True)
    filtered_condition_assembly = condition_assembly.where(mask, drop=True)
    return filtered_condition_assembly, filtered_baseline_assembly
