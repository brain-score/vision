import numpy as np
from brainscore_core import Metric
from tqdm import tqdm
from typing import Dict

from brainscore_vision import load_dataset, load_stimulus_set
from brainio.assemblies import BehavioralAssembly
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics import Score
from brainscore_vision.metrics.value_delta import ValueDelta
from brainscore_vision.model_interface import BrainModel
from .helpers.helpers import generate_summary_df, calculate_integral, HUMAN_INTEGRAL_ERRORS, LAPSE_RATES, \
    split_dataframe

BIBTEX = """TBD"""

DATASETS = ['circle_line', 'color', 'convergence', 'eighth',
            'gray_easy', 'gray_hard', 'half', 'juncture',
            'lle', 'llh', 'quarter', 'round_f',
            'round_v', 'tilted_line']

PRECOMPUTED_CEILINGS = {'circle_line': [0.883, 0.078], 'color': [0.897, 0.072], 'convergence': [0.862, 0.098],
                        'eighth': [0.852, 0.107], 'gray_easy': [0.907, 0.064], 'gray_hard': [0.863, 0.099],
                        'half': [0.898, 0.077], 'juncture': [0.767, 0.141], 'lle': [0.831, 0.116], 'llh': [0.812, 0.123],
                        'quarter': [0.876, 0.087], 'round_f': [0.874, 0.088], 'round_v': [0.853, 0.101],
                        'tilted_line': [0.912, 0.064]}

for dataset in DATASETS:
    identifier = f"Ferguson2024{dataset}ValueDelta"
    globals()[identifier] = lambda dataset=dataset: _Ferguson2024ValueDelta(dataset)


class _Ferguson2024ValueDelta(BenchmarkBase):
    def __init__(self, dataset, precompute_ceiling=False):
        self._metric = ValueDelta(scale=0.75)  # 0.75 chosen after calibrating with ceiling
        self._fitting_stimuli = load_stimulus_set(f'Ferguson2024_{dataset}')
        self._assembly = load_dataset(f'Ferguson2024_{dataset}')
        self._visual_degrees = 8
        self._number_of_trials = 3
        self._ceiling = calculate_ceiling(precompute_ceiling, dataset, self._assembly, self._metric, num_loops=500)
        super(_Ferguson2024ValueDelta, self).__init__(identifier="Ferguson2024", version=1, ceiling_func=self._ceiling,
                                                      parent='behavior',
                                                      bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel) -> Score:
        # fitting_stimuli = place_on_screen(self._fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
        #                                   source_visual_degrees=self._visual_degrees)
        # candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        # stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
        #                                source_visual_degrees=self._visual_degrees)
        # probabilities = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)

        human_results = get_human_integral_data(self._assembly, dataset)
        human_integral = human_results["human_integral"]
        model_integral = 1.5
        raw_score = self._metric(model_integral, human_integral)
        ceiling = self._ceiling
        score = min(max(raw_score / ceiling, 0), 1)  # ensure ceiled score is between 0 and 1
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score


def Ferguson2024ValueDelta(experiment):
    return _Ferguson2024ValueDelta(experiment)


def calculate_ceiling(precompute_ceiling, dataset: str, assembly: BehavioralAssembly, metric: Metric,
                      num_loops: int) -> Score:
    """
    - A Version of split-half reliability, in which the data is split randomly in half
     and the metric is called on those two halves.

    :param dataset: str, the prefix of the experiment subtype, ex: "tilted_line" or "lle"
    :param assembly: the human behavioral data to look at
    :param metric: of type Metric, used to calculate the score between two subjects
    :return: Score object consisting of the score between two halves of human data
    :param num_loops: int. number of times the score is calculated. Final score is the average of all of these.
    """
    if precompute_ceiling:
        score = Score(PRECOMPUTED_CEILINGS[dataset][0])
        score.attrs['error'] = 0.054
        score.attrs[Score.RAW_VALUES_KEY] = [0.8832, 0.054]
        return score
    else:
        scores = []
        for i in tqdm(range(num_loops)):
            half_1, half_2 = split_dataframe(assembly, seed=i)
            half_1_score = get_human_integral_data(half_1, dataset)["human_integral"]
            half_2_score = get_human_integral_data(half_2, dataset)["human_integral"]
            score = metric(half_1_score, half_2_score)
            scores.append(score)

        score = Score(np.mean(scores))
        scores = np.array(scores, dtype=float)
        score.attrs['error'] = np.std(scores)
        score.attrs[Score.RAW_VALUES_KEY] = [np.mean(scores), np.std(scores)]
        print(f"Dataset: {dataset}, score: {np.mean(scores)}, error: {score.attrs['error']}")
        return score


def get_human_integral_data(assembly: BehavioralAssembly, dataset: str) -> Dict:
    """
    - Generates summary data for the experiment and calculates the human integral of delta line

    :param assembly: the human behavioral data to look at
    :param dataset: str, the prefix of the experiment subtype, ex: "tilted_line" or "lle"
    :return: tuple of calculated human integral and its boostrapped (precomputed) error
    """
    lapse_rate = LAPSE_RATES[dataset]
    blue_data = generate_summary_df(assembly, lapse_rate, "first")
    orange_data = generate_summary_df(assembly, lapse_rate, "second")
    human_integral = calculate_integral(blue_data, orange_data)
    human_integral_error = HUMAN_INTEGRAL_ERRORS[dataset]
    return dict(zip(["human_integral", "human_integral_error"], [human_integral, human_integral_error]))
