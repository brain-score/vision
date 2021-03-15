import numpy as np

import brainscore
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.ost import OSTCorrelation
from brainscore.model_interface import BrainModel

BIBTEX = """@Article{Kar2019,
            author={Kar, Kohitij
            and Kubilius, Jonas
            and Schmidt, Kailyn
            and Issa, Elias B.
            and DiCarlo, James J.},
            title={Evidence that recurrent circuits are critical to the ventral stream's execution of core object recognition behavior},
            journal={Nature Neuroscience},
            year={2019},
            month={Jun},
            day={01},
            volume={22},
            number={6},
            pages={974-983},
            abstract={Non-recurrent deep convolutional neural networks (CNNs) are currently the best at modeling core object recognition, a behavior that is supported by the densely recurrent primate ventral stream, culminating in the inferior temporal (IT) cortex. If recurrence is critical to this behavior, then primates should outperform feedforward-only deep CNNs for images that require additional recurrent processing beyond the feedforward IT response. Here we first used behavioral methods to discover hundreds of these `challenge' images. Second, using large-scale electrophysiology, we observed that behaviorally sufficient object identity solutions emerged {\textasciitilde}30{\thinspace}ms later in the IT cortex for challenge images compared with primate performance-matched `control' images. Third, these behaviorally critical late-phase IT response patterns were poorly predicted by feedforward deep CNN activations. Notably, very-deep CNNs and shallower recurrent CNNs better predicted these late IT responses, suggesting that there is a functional equivalence between additional nonlinear transformations and recurrence. Beyond arguing that recurrent circuits are critical for rapid object identification, our results provide strong constraints for future recurrent model development.},
            issn={1546-1726},
            doi={10.1038/s41593-019-0392-5},
            url={https://doi.org/10.1038/s41593-019-0392-5}
            }"""
VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 44
TIME_BINS = [(time_bin_start, time_bin_start + 10) for time_bin_start in range(70, 250, 10)]


class DicarloKar2019OST(BenchmarkBase):
    def __init__(self):
        ceiling = Score([.79, np.nan],  # following private conversation with Kohitij Kar
                        coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(DicarloKar2019OST, self).__init__(identifier='dicarlo.Kar2019-ost', version=2,
                                                ceiling_func=lambda: ceiling,
                                                parent='IT',
                                                bibtex=BIBTEX)
        assembly = brainscore.get_assembly('dicarlo.Kar2019')
        # drop duplicate images
        _, index = np.unique(assembly['image_id'], return_index=True)
        assembly = assembly.isel(presentation=index)
        assembly.attrs['stimulus_set'] = assembly.stimulus_set.drop_duplicates('image_id')

        assembly = assembly.sel(decoder='svm')

        self._assembly = assembly
        self._assembly['truth'] = self._assembly['image_label']
        self._assembly.stimulus_set['truth'] = self._assembly.stimulus_set['image_label']

        self._similarity_metric = OSTCorrelation()

        self._visual_degrees = VISUAL_DEGREES
        self._number_of_trials = NUMBER_OF_TRIALS
        self._time_bins = TIME_BINS

    def __call__(self, candidate: BrainModel):
        candidate.start_recording('IT', time_bins=self._time_bins)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        # Temporal recordings from large candidates take up a lot of memory and compute time.
        # In order to quickly reject recordings that are static over time,
        # we will show one image and check whether the recordings vary over time at all or not.
        # If they don't we can quickly score the candidate with a failure state
        # since it will not be able to predict temporal differences with the OST metric
        check_recordings = candidate.look_at(stimulus_set[:1], number_of_trials=self._number_of_trials)
        if not temporally_varying(check_recordings):
            return Score([np.nan, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])

        recordings = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        score = self._similarity_metric(recordings, self._assembly)
        score = ceil_score(score, self.ceiling)
        return score


def temporally_varying(recordings):
    """
    Tests whether the given recordings change over time, for any of the stimuli on any of the neuroids

    :return True if any of the neuroids changes over time for any of the stimuli, False otherwise
    """
    recordings = recordings.transpose('presentation', 'neuroid', 'time_bin')
    first_response = recordings.sel(time_bin=recordings['time_bin'].values[0])
    different = recordings != first_response
    return different.any()
