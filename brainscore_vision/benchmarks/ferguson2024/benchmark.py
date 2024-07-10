import numpy as np
from brainscore_core import Metric
from brainio.stimuli import StimulusSet
from tqdm import tqdm
from typing import Dict
import xarray as xr
from brainscore_vision import load_dataset, load_stimulus_set, load_metric
from brainio.assemblies import BehavioralAssembly
from brainscore_vision.benchmark_helpers.screen import place_on_screen
import pandas as pd
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics import Score
from brainscore_vision.model_interface import BrainModel
from .helpers.helpers import generate_summary_df, calculate_integral, HUMAN_INTEGRAL_ERRORS, LAPSE_RATES, \
    split_dataframe, boostrap_integral

BIBTEX = """
        @misc{ferguson_ngo_lee_dicarlo_schrimpf_2024,
         title={How Well is Visual Search Asymmetry predicted by a Binary-Choice, Rapid, Accuracy-based Visual-search, Oddball-detection (BRAVO) task?},
         url={osf.io/5ba3n},
         DOI={10.17605/OSF.IO/5BA3N},
         publisher={OSF},
         author={Ferguson, Michael E, Jr and Ngo, Jerry and Lee, Michael and DiCarlo, James and Schrimpf, Martin},
         year={2024},
         month={Jun}
}
"""

# These ceilings were precomputed to save time in benchmark execution
PRECOMPUTED_CEILINGS = {'circle_line': [0.883, 0.078], 'color': [0.897, 0.072], 'convergence': [0.862, 0.098],
                        'eighth': [0.852, 0.107], 'gray_easy': [0.907, 0.064], 'gray_hard': [0.863, 0.099],
                        'half': [0.898, 0.077], 'juncture': [0.767, 0.141], 'lle': [0.831, 0.116], 'llh': [0.812, 0.123],
                        'quarter': [0.876, 0.087], 'round_f': [0.874, 0.088], 'round_v': [0.853, 0.101],
                        'tilted_line': [0.912, 0.064]}


for dataset in PRECOMPUTED_CEILINGS.keys():
    identifier = f"Ferguson2024{dataset}ValueDelta"
    globals()[identifier] = lambda dataset=dataset: _Ferguson2024ValueDelta(dataset)


class _Ferguson2024ValueDelta(BenchmarkBase):
    def __init__(self, experiment, precompute_ceiling=True):
        self._experiment = experiment
        self._precompute_ceiling = precompute_ceiling
        self._metric = load_metric('value_delta', scale=0.75)  # 0.75 chosen after calibrating with ceiling
        self._fitting_stimuli = gather_fitting_stimuli(combine_all=False, experiment=self._experiment)
        self._assembly = load_dataset(f'Ferguson2024_{self._experiment}')
        self._visual_degrees = 8
        self._number_of_trials = 3
        self._ceiling = calculate_ceiling(self._precompute_ceiling, self._experiment, self._assembly, self._metric, num_loops=500)
        super(_Ferguson2024ValueDelta, self).__init__(identifier=f"Ferguson2024{self._experiment}-value_delta",
                                                      version=1, ceiling_func=self._ceiling,
                                                      parent='behavior', bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel) -> Score:

        # add truth labels to stimuli and training data
        self._assembly.stimulus_set["image_label"] = np.where(self._assembly.stimulus_set["image_number"] % 2 == 0, "oddball", "same")
        self._fitting_stimuli["image_label"] = np.where(self._fitting_stimuli["image_number"] % 2 == 0, "oddball", "same")

        # fit logistic binary decoder and perform task:
        fitting_stimuli = place_on_screen(self._fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                           source_visual_degrees=self._visual_degrees)
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        model_labels_raw = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        model_labels = process_model_choices(model_labels_raw)
        human_integral = get_integral_data(self._assembly, self._experiment)['integral']
        model_integral = get_integral_data(model_labels, self._experiment)['integral']
        raw_score = self._metric(model_integral, human_integral)
        ceiling = self._ceiling
        score = Score(min(max(raw_score / ceiling, 0), 1))  # ensure ceiled score is between 0 and 1
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

    :param precompute_ceiling: True if using precomputed ceilings. Should almost always be True.
    :param dataset: str, the prefix of the experiment subtype, ex: "tilted_line" or "lle"
    :param assembly: the human behavioral data to look at
    :param metric: of type Metric, used to calculate the score between two subjects
    :return: Score object consisting of the score between two halves of human data
    :param num_loops: int. number of times the score is calculated. Final score is the average of all of these.
    """
    if precompute_ceiling:
        score = Score(PRECOMPUTED_CEILINGS[dataset][0])
        score.attrs['error'] = PRECOMPUTED_CEILINGS[dataset][1]
        score.attrs[Score.RAW_VALUES_KEY] = [PRECOMPUTED_CEILINGS[dataset][0], PRECOMPUTED_CEILINGS[dataset][1]]
        return score
    else:
        scores = []
        for i in tqdm(range(num_loops)):
            half_1, half_2 = split_dataframe(assembly, seed=i)
            half_1_score = get_integral_data(half_1, dataset)["integral"]
            half_2_score = get_integral_data(half_2, dataset)["integral"]
            score = metric(half_1_score, half_2_score)
            scores.append(score)

        score = Score(np.mean(scores))
        scores = np.array(scores, dtype=float)
        score.attrs['error'] = np.std(scores)
        score.attrs[Score.RAW_VALUES_KEY] = [np.mean(scores), np.std(scores)]
        print(f"Dataset: {dataset}, score: {np.mean(scores)}, error: {score.attrs['error']}")
        return score


def get_integral_data(assembly: BehavioralAssembly, experiment: str, precompute_boostrap=True) -> Dict:
    """
    - Generates summary data for the experiment and calculates the integral of delta line

    :param assembly: the human behavioral data to look at
    :param experiment: str, the prefix of the experiment subtype, ex: "tilted_line" or "lle"
    :param precompute_boostrap: True if using precomputed integral errors, else manually compute (Slow!)
    :return: tuple of calculated human integral and its boostrapped (precomputed) error
    """
    lapse_rate = LAPSE_RATES[experiment]
    blue_data = generate_summary_df(assembly, lapse_rate, "first")
    orange_data = generate_summary_df(assembly, lapse_rate, "second")
    integral = calculate_integral(blue_data, orange_data)
    integral_error = HUMAN_INTEGRAL_ERRORS[experiment] if precompute_boostrap else \
        boostrap_integral(blue_data, orange_data)["integral_std"]
    return dict(zip(["integral", "integral_error"], [integral, integral_error]))


def gather_fitting_stimuli(combine_all=True, experiment="") -> StimulusSet:
    """
    Combines all the training stimuli into one merged stimulus_set, or returns the selected set for the experiment

    :param combine_all: True if you want to collapse all 14 stimuli pairs into 1 stimulus set
    :param experiment: only if combine_all is False, then specify which experiment's stimuli you want
    :return: merged StimulusSet of all 14 stimulus set's training data or a StimulusSet of that experiment only
    """

    if combine_all:
        all_stimulus_sets = []
        for experiment in PRECOMPUTED_CEILINGS.keys():
            stimulus_set = load_stimulus_set(f"Ferguson2024_{experiment}_training_stimuli")
            all_stimulus_sets.append(stimulus_set)
        merged_dataframe = pd.concat(all_stimulus_sets, axis=0, ignore_index=True)
        merged_dataframe.name = "Ferguson2024_merged_training_stimuli"
        merged_dataframe.identifier = "Ferguson2024_merged_training_stimuli"
        merged_stimulus_set = StimulusSet(merged_dataframe)
        merged_stimulus_set.identifier = "Ferguson2024_merged_training_stimuli"
        return merged_stimulus_set
    else:
        return load_stimulus_set(f"Ferguson2024_{experiment}_training_stimuli")


def process_model_choices(raw_model_labels: BehavioralAssembly) -> BehavioralAssembly:
    """
    Takes in a raw Assembly and applies a softmax and threshold to get a string label for a class. Also
    builds the model's assembly to resemble a humans by adding fields (trial_type, num_distractors, etc)

    :param raw_model_labels: a BehavioralAssembly that has two raw values corresponding to class choices
    :return: new assembly with an added dim, "model_choice" based on the raw values
    """
    distractor_mapping = {
        1.0: [0, 1, 6, 7, 12, 13, 18, 19],
        5.0: [2, 3, 8, 9, 14, 15, 20, 21]
    }
    distractor_lookup = {image_num: str(distractor) for distractor, images in distractor_mapping.items() for image_num
                         in images}

    def num_distractors(image_num):
        return distractor_lookup.get(image_num, "11.0")  # default to 11.0 if not found, which should never happen

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def is_even(image_num):
        return 1 if image_num % 2 == 0 else 0

    def is_equal(label, image_type):
        return 1 if label == image_type else 0

    softmax_values = softmax(raw_model_labels.values)
    labels = np.where(softmax_values[:, 0] > 0.5, 1, 0)
    labels_string = np.where(softmax_values[:, 0] > 0.5, "oddball", "same")
    model_choices = xr.DataArray(labels, dims=["presentation"], coords={"presentation": raw_model_labels.coords["presentation"]})
    model_choices = model_choices.assign_coords(trial_type=('presentation', np.array(['normal'] * 48)))
    model_choices = model_choices.assign_coords(participant_id=('presentation', np.array(['model'] * 48)))
    distractor_nums = [num_distractors(image_num) for image_num in model_choices.coords['image_number'].values]
    model_choices = model_choices.assign_coords(distractor_nums=('presentation', distractor_nums))
    target_present = [is_even(image_num) for image_num in model_choices.coords['image_number'].values]
    model_choices = model_choices.assign_coords(target_present=('presentation', target_present))
    model_choices = model_choices.assign_coords(labels_string=('presentation', labels_string))
    correct = [
        is_equal(label, image_type)
        for label, image_type in zip(
            model_choices.coords['labels_string'].values,
            model_choices.coords['image_label'].values
        )
    ]
    model_choices = model_choices.assign_coords(correct=('presentation', correct))
    return BehavioralAssembly(model_choices)


