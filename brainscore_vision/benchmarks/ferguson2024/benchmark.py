import numpy as np
from brainscore_core import Metric
from tqdm import tqdm

from brainio.assemblies import walk_coords
from brainscore_vision import load_dataset, load_metric, load_stimulus_set
from brainio.assemblies import BehavioralAssembly
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics import Score
from brainscore_vision.metrics.value_delta import ValueDelta
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import LazyLoad
import xarray as xr
from .helpers.helpers import generate_summary_df, calculate_integral, HUMAN_INTEGRAL_ERRORS, LAPSE_RATES

BIBTEX = """TBD"""

DATASETS = ['circle_line', 'color', 'convergence', 'eighth',
            'gray_easy', 'gray_hard', 'half', 'juncture',
            'lle', 'llh', 'quarter', 'round_f',
            'round_v', 'tilted_line']

for dataset in DATASETS:
    # behavioral benchmark
    identifier = f"Ferguson2024{dataset}AlignmentMeasure"
    globals()[identifier] = lambda dataset=dataset: _Ferguson2024ValueDelta(dataset)


class _Ferguson2024ValueDelta(BenchmarkBase):
    def __init__(self, dataset, precompute_ceiling=False):
        self._metric = ValueDelta(scale=0.75)  # 0.75 chosen after calibrating with ceiling
        self._fitting_stimuli = load_stimulus_set(f'Ferguson2024_{dataset}')
        self._assembly = load_dataset(f'Ferguson2024_{dataset}')
        self._visual_degrees = 8
        self._number_of_trials = 3
        self._ceiling = 0.0885 if precompute_ceiling is True else calculate_ceiling(dataset, self._assembly,
                                                                                    self._metric, num_loops=500)
        super(_Ferguson2024ValueDelta, self).__init__(
            identifier="Ferguson2024", version=2,
            ceiling_func=self._ceiling,
            parent='behavior',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel) -> Score:
        # fitting_stimuli = place_on_screen(self._fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
        #                                   source_visual_degrees=self._visual_degrees)
        # candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        # stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
        #                                source_visual_degrees=self._visual_degrees)
        # probabilities = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)

        human_integral, human_integral_error = get_human_integral_data(self._assembly, dataset)
        # model_integral = accuracy_metric(ground_truth, probabilities)
        # score = self._metric(model_accuracy, human_accuracy)
        ceiling = self.ceiling
        #score = self.ceil_score(1, ceiling)
        return score


def Ferguson2024ValueDelta(experiment):
    return _Ferguson2024ValueDelta(experiment)


def calculate_ceiling(dataset: str, assembly: BehavioralAssembly, metric: Metric, num_loops: int) -> Score:
    """
    A Version of split-half reliability, in which the data is split randomly in half
    and the metric is called on those two halves.

    :param dataset: str, the prefix of the experiment subtype, ex: "tilted_line" or "lle"
    :param assembly: the human behavioral data to look at
    :param metric: of type Metric, used to calculate the score between two subjects
    :return: Score object consisting of the score between two halfs of hman data
    :param num_loops: int. number of times the score is calculated. Final score is the average of all of these.
    """

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
    return score


def split_dataframe(df: BehavioralAssembly, seed: int) -> (BehavioralAssembly, BehavioralAssembly):
    """
    Takes in one DF and splits it into two, randomly, on the presentation dim

    :param df: The DataFrame (assembly) to split
    :param seed: a seed for the numpy rng
    :return: Two DataFrames (assemblies)
    """
    if seed is not None:
        np.random.seed(seed)
    shuffled_indices = np.random.permutation(df.presentation.size)
    half = len(shuffled_indices) // 2
    indices_1 = shuffled_indices[:half]
    indices_2 = shuffled_indices[half:]
    dataarray_1 = df.isel(presentation=indices_1)
    dataarray_2 = df.isel(presentation=indices_2)
    return dataarray_1, dataarray_2


def get_human_integral_data(assembly: BehavioralAssembly, dataset: str) -> (float, float):
    """
    Generates summary data for the experiment and calculates the human integral of delta line

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
