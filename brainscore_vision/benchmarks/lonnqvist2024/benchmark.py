from pathlib import Path

import numpy as np

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import load_metric, load_stimulus_set, load_dataset
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics import Score
from brainscore_vision.model_interface import BrainModel

BIBTEX = ""  # to appear in a future article


class _Lonnqvist2024Base(BenchmarkBase):
    def __init__(self, identifier, dataset, ceiling_func, metric):
        self._metric = metric
        self._stimulus_set = load_stimulus_set('Lonnqvist2024_test')
        self._fitting_stimuli = load_stimulus_set('Lonnqvist2024_train')
        self._visual_degrees = 17.70753
        self.assembly = load_dataset(f'Lonnqvist2024_{dataset}')

        super(_Lonnqvist2024Base, self).__init__(
            identifier=identifier, version=1,
            ceiling_func=ceiling_func,
            parent='Lonnqvist2024',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel, return_raw_responses: bool = False):
        fitting_stimulus_set = place_on_screen(
            self._fitting_stimuli,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees
        )
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli=fitting_stimulus_set, number_of_trials=1)
        stimulus_set = place_on_screen(
            self._stimulus_set,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees
        )
        model_response = candidate.look_at(stimulus_set, number_of_trials=1)
        model_response = convert_proba_to_choices(model_response)
        raw_score = self._metric(model_response, self.assembly)
        # Adjust score to ceiling
        ceiling = self.ceiling
        score = raw_score / ceiling
        # ensure score <= 1.0
        if score.values > 1:
            score = Score(np.array(1.))
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        if return_raw_responses:
            return score, model_response
        return score


class _Lonnqvist2024BehavioralAccuracyDistanceInlabInstructions(_Lonnqvist2024Base):
    def __init__(self):
        metric = load_metric('accuracy_distance')
        ceiling_func = lambda: metric.ceiling(self.assembly)
        super(_Lonnqvist2024BehavioralAccuracyDistanceInlabInstructions, self).__init__(
            identifier='Lonnqvist2024-inlab-instructions_behavioral_accuracy_distance', dataset='inlab-instructions',
            ceiling_func=ceiling_func,
            metric=metric)


class _Lonnqvist2024BehavioralAccuracyDistanceInlabNoInstructions(_Lonnqvist2024Base):
    def __init__(self):
        metric = load_metric('accuracy_distance')
        ceiling_func = lambda: metric.ceiling(self.assembly)
        super(_Lonnqvist2024BehavioralAccuracyDistanceInlabNoInstructions, self).__init__(
            identifier='Lonnqvist2024-inlab-no-instructions_behavioral_accuracy_distance', dataset='inlab-no-instructions',
            ceiling_func=ceiling_func,
            metric=metric)


class _Lonnqvist2024BehavioralAccuracyDistanceOnlineNoInstructions(_Lonnqvist2024Base):
    def __init__(self):
        metric = load_metric('accuracy_distance')
        ceiling_func = lambda: metric.ceiling(self.assembly)
        super(_Lonnqvist2024BehavioralAccuracyDistanceOnlineNoInstructions, self).__init__(
            identifier='Lonnqvist2024-online-no-instructions_behavioral_accuracy_distance', dataset='online-no-instructions',
            ceiling_func=ceiling_func,
            metric=metric)


class _Lonnqvist2024EngineeringAccuracy(_Lonnqvist2024Base):
    def __init__(self):
        metric = load_metric('accuracy')
        ceiling_func = lambda: Score(1)
        super(_Lonnqvist2024EngineeringAccuracy, self).__init__(
            identifier='Lonnqvist2024-engineering_accuracy', dataset='inlab-instructions',
            ceiling_func=ceiling_func,
            metric=metric)

    def __call__(self, candidate: BrainModel, return_raw_responses: bool = False):
        fitting_stimulus_set = place_on_screen(
            self._fitting_stimuli,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees
        )
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli=fitting_stimulus_set, number_of_trials=1)
        stimulus_set = place_on_screen(
            self._stimulus_set,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees
        )
        model_response = candidate.look_at(stimulus_set, number_of_trials=1)
        model_response = convert_proba_to_choices(model_response)
        raw_score = self._metric(model_response, stimulus_set['truth'])
        # Adjust score to ceiling
        ceiling = self.ceiling
        score = raw_score / ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        if return_raw_responses:
            return score, model_response
        return score


def convert_proba_to_choices(source: BehavioralAssembly) -> np.array:
    """Converts the probability values returned by models doing probability tasks to behavioral choices."""
    decisions = np.argmax(source.values, axis=1)
    choices = [source['choice'].values[decision] for decision in decisions]
    return BehavioralAssembly(choices, coords={'presentation': source['presentation']})
