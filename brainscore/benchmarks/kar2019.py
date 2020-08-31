import numpy as np

import brainscore
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.screen import place_on_screen
from brainscore.benchmarks.trials import repeat_trials, average_trials
from brainscore.metrics import Score
from brainscore.metrics.ost import OSTCorrelation
from brainscore.model_interface import BrainModel

VISUAL_DEGREES = 8


class DicarloKar2019OST(BenchmarkBase):
    def __init__(self):
        ceiling = Score([.79, np.nan],  # following private conversation with Kohitij Kar
                        coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(DicarloKar2019OST, self).__init__(identifier='dicarlo.Kar2019-ost', version=2,
                                                ceiling_func=lambda: ceiling,
                                                parent='IT-temporal',
                                                bibtex= """@Article{Kar2019,
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
                                                    }""")
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
        self._number_of_trials = 44


    def __call__(self, candidate: BrainModel):
        time_bins = [(time_bin_start, time_bin_start + 10) for time_bin_start in range(70, 250, 10)]
        candidate.start_recording('IT', time_bins=time_bins)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        stimulus_set = repeat_trials(stimulus_set, number_of_trials=self._number_of_trials)
        recordings = candidate.look_at(stimulus_set)
        recordings = average_trials(recordings)
        score = self._similarity_metric(recordings, self._assembly)
        score = ceil_score(score, self.ceiling)
        return score
