import numpy as np

import brainscore
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.ost import OSTCorrelation
from brainscore.model_interface import BrainModel

VISUAL_DEGREES = 8


class DicarloKar2019OST(BenchmarkBase):
    def __init__(self):
        ceiling = Score([.79, np.nan],  # following private conversation with Kohitij Kar
                        coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(DicarloKar2019OST, self).__init__(identifier='dicarlo.Kar2019-ost', parent='IT', version=2,
                                                ceiling_func=lambda: ceiling,
                                                bibtex=
                                                """@article{kar2019evidence,
                                                  title={Evidence that recurrent circuits are critical to the ventral stream’s execution of core object recognition behavior},
                                                  author={Kar, Kohitij and Kubilius, Jonas and Schmidt, Kailyn and Issa, Elias B and DiCarlo, James J},
                                                  journal={Nature neuroscience},
                                                  volume={22},
                                                  number={6},
                                                  pages={974--983},
                                                  year={2019},
                                                  url={https://www.nature.com/articles/s41593-019-0392-5},
                                                  publisher={Nature Publishing Group}
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

    def __call__(self, candidate: BrainModel):
        time_bins = [(time_bin_start, time_bin_start + 10) for time_bin_start in range(70, 250, 10)]
        candidate.start_recording('IT', time_bins=time_bins)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        recordings = candidate.look_at(stimulus_set)
        score = self._similarity_metric(recordings, self._assembly)
        score = ceil_score(score, self.ceiling)
        return score
