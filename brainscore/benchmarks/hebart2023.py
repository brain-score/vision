import numpy as np

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad



BIBTEX = """@article{10.7554/eLife.82580,
          author = {Hebart, Martin N and Contier, Oliver and Teichmann, Lina and Rockter, Adam H and Zheng, Charles Y and Kidder, Alexis and Corriveau, Anna and Vaziri-Pashkam, Maryam and Baker, Chris I},
          journal = {eLife},
          month = {feb},
          pages = {e82580},
          title = {THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior},
          volume = 12,
          year = 2023
          }"""

class Hebart2023Accuracy(BenchmarkBase):
    def __init__(self, similarity_measure):
        self._metric = None
        self._visual_degrees = None  # Can we know this, we collected the data using mturk?
        self._number_of_trials = None
        self.validation_data = None  # https://osf.io/b2a4j -> host on S3 
        self.ceiling = 0.6844

        #  What is the difference here? What should I use?
        self._assembly = LazyLoad(lambda: load_assembly("Hebart2023"))     # <- this is from Geirhos er al.
        def load_assembly(dataset):                                        # 
            pass                                                           #
        
        self.stimulus_set  = brainscore.get_stimulus_set("Hebart2023")     # <- this is from Islam2021
        
        super(Hebart2023Accuracy, self).__init__(
            identifier=f'Hebart2023Accuracy_{similarity_measure}', version=1,
            ceiling_func=0.6844,                                             # Will need the noise ceiling from Martin H.
            parent='Hebart2023',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        candidate.start_task(BrainModel.Task.odd_one_out, self.similarity_measure) # Is it ok to pass the similarity_measure here?
        predicted_odd_one_outs = candidate.look_at(self._assembly.stimulus_set, number_of_trials=self._number_of_trials)
        raw_score = self._metric(predicted_odd_one_outs, self._assembly.validation_data)
        ceiling = self.ceiling
        score = (raw_score - 1/3) / (ceiling - 1/3)
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score
    

