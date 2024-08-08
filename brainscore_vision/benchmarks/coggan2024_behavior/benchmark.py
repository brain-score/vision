# Created by David Coggan on 2024 06 25

import numpy as np
from brainio.assemblies import DataAssembly, BehavioralAssembly
from brainscore_vision import load_stimulus_set, load_dataset
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_core.metrics import Score
from brainscore_vision.metric_helpers import Defaults as XarrayDefaults
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import LazyLoad
from scipy.stats import sem
import pandas as pd

# the BIBTEX will be used to link to the publication from the benchmark for further details
BIBTEX = """@article {
    Tong.Coggan2024.behavior,
    author = {David D. Coggan and Frank Tong},
    title = {Modeling human visual recognition of occluded objects}},
    year = {2024},
    url = {},
    journal = {in prep}}"""

class Coggan2024_behavior_ConditionWiseAccuracySimilarity(BenchmarkBase):

    """
    This benchmark measures classification accuracy for a set of occluded object images, then attains the mean accuracy
    for each of the 18 occlusion conditions. This is then correlated with the corresponding accuracies for each of the
    30 human subjects in the behavioral experiment to obtain the brain score.
    Note: Because the object-occluder pairings were randomized for each subject, image-level metrics (e.g., error
    consistency) have limited utility here as a ceiling cannot be calculated.
    """

    def __init__(self):
        self._fitting_stimuli = load_stimulus_set('Coggan2024_behavior_fitting')  # this fails is wrapped by LazyLoad
        self._assembly = LazyLoad(lambda: load_dataset('Coggan2024_behavior'))
        self._visual_degrees = 10
        self._number_of_trials = 1
        self._ceiling_func = lambda assembly: get_noise_ceiling(assembly)
        super(Coggan2024_behavior_ConditionWiseAccuracySimilarity, self).__init__(
            identifier='tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity',
            version=1,
            ceiling_func=lambda df: get_noise_ceiling(df),
            parent='behavior',
            bibtex=BIBTEX,
        )

    def __call__(self, candidate: BrainModel) -> Score:

        fitting_stimuli = place_on_screen(
            self._fitting_stimuli,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees)
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        stimulus_set = place_on_screen(
            self._assembly.stimulus_set,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees)
        probabilities = candidate.look_at(
            stimulus_set, number_of_trials=self._number_of_trials)
        model_predictions = [
            probabilities.choice[c].values for c in probabilities.argmax(axis=1)]

        data = pd.DataFrame(dict(
            subject=self._assembly.subject,
            object_class=self._assembly.object_class,
            occluder_type=self._assembly.occluder_type,
            occluder_color=self._assembly.occluder_color,
            visibility=self._assembly.visibility,
            human_prediction=self._assembly.values,
            human_accuracy=self._assembly.human_accuracy,
            model_prediction=model_predictions
        ))
        data['model_accuracy'] = pd.Series(
            data.model_prediction == data.object_class, dtype=int)

        # get correlation between model and human performance across conditions
        performance = (
            data[data.visibility < 1]
            .groupby(['subject', 'occluder_type', 'occluder_color'])
            .mean(numeric_only=True)
            .reset_index()
        )
        scores = performance.groupby('subject').apply(
            lambda df: np.corrcoef(df.human_accuracy, df.model_accuracy)[0, 1])
        score = Score(np.mean(scores))
        score.attrs['raw'] = scores

        # get ceiled score
        ceiled_score = ceiler(score, self._ceiling_func(performance))
        ceiled_score.attrs['raw'] = score

        return ceiled_score


def get_noise_ceiling(performance: pd.DataFrame) -> Score:
    """
    Returns the noise ceiling for human similarity estimates. This is the lower bound of typical noise-ceiling range
    (e.g. Nili et al., 2014), i.e., the correlation of condition-wise accuracies between each individual subject and
    the mean of the remaining subjects in the sample. This matches how the model is scored, if the group values are
    substituted for model values.
    """
    nc = []
    for subject in performance.subject.unique():
        performance_ind = performance[performance.subject == subject]
        performance_grp = performance[performance.subject != subject]
        numeric_cols = performance_grp.select_dtypes(include=np.number).columns
        performance_grp = performance_grp.groupby(['occluder_type', 'occluder_color'])[numeric_cols].mean()
        merged_df = performance_ind.merge(
            performance_grp, on=['occluder_type', 'occluder_color'])
        nc.append(np.corrcoef(merged_df.human_accuracy_x, merged_df.human_accuracy_y)[0, 1])
    ceiling = Score(np.mean(nc))
    ceiling.attrs['raw'] = nc
    ceiling.attrs['error'] = sem(nc)
    return ceiling


def ceiler(score: Score, ceiling: Score) -> Score:
    # ro(X, Y)
    # = (r(X, Y) / sqrt(r(X, X) * r(Y, Y)))^2
    # = (r(X, Y) / sqrt(r(Y, Y) * r(Y, Y)))^2  # assuming that r(Y, Y) ~ r(X, X) following Yamins 2014
    # = (r(X, Y) / r(Y, Y))^2
    r_square = np.power(score.values / ceiling.values, 2)
    ceiled_score = Score(r_square)
    if 'error' in score.attrs:
        ceiled_score.attrs['error'] = score.attrs['error']
    ceiled_score.attrs[Score.RAW_VALUES_KEY] = score
    ceiled_score.attrs['ceiling'] = ceiling
    return ceiled_score




