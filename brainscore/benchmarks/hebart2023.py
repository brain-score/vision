import numpy as np

import brainscore
from brainio.assemblies import walk_coords
from brainscore.benchmarks import BenchmarkBase
from brainscore.model_interface import BrainModel
from brainscore.metrics import Score
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
        self._metric = 'None'
        self._visual_degrees = 6
        self._number_of_trials = 1
        self.ceiling = 0.6844

        super(Hebart2023Accuracy, self).__init__(
            identifier=f'Hebart2023Accuracy_{similarity_measure}', version=1,
            ceiling_func=lambda: Score([0.6844, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation']),
            parent='Hebart2023',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        candidate.start_task(BrainModel.Task.odd_one_out, None) 
        predicted_odd_one_outs = candidate.look_at(self._assembly.stimulus_set, number_of_trials=self._number_of_trials)
        raw_score = self._metric(predicted_odd_one_outs, self._assembly.validation_data)
        ceiling = self.ceiling
        score = (raw_score - 1/3) / (ceiling - 1/3)
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score
    
    def load_assembly(self):
        assembly = brainscore.get_assembly(f'Hebart2023') 
        stimulus_set = assembly.attrs['stimulus_set']
        stimulus_set = stimulus_set[stimulus_set['stimulus_id'].isin(set(assembly['stimulus_id'].values))]
        assembly.attrs['stimulus_set'] = stimulus_set
        return assembly

    """
    def cast_coordinate_type(self, assembly, coordinate, newtype):
        #TODO: This might need to be adapted to out data
        attrs = assembly.attrs
        condition_values = assembly[coordinate].values
        assembly = type(assembly)(assembly.values, coords={
          coord: (dims, values) for coord, dims, values in walk_coords(assembly) if coord != coordinate},
                                  dims=assembly.dims)
        assembly[coordinate] = 'presentation', condition_values.astype(newtype)
        assembly = type(assembly)(assembly)
        assembly.attrs = attrs
        return assembly
    """


    

