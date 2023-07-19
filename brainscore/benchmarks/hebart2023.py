import numpy as np
from brainscore.benchmarks import BenchmarkBase
from brainscore.model_interface import BrainModel


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
    def __init__(self):
        self._metric = None
        self._assembly = None
        self._visual_degrees = None

        self._number_of_trials = None
        
        #TODO I am not sure where this goes , but it should be in the benchmark right?
        self.validation_data = None  # https://osf.io/b2a4j we'll need to convert this to BehavioralAssembly right?

        # TODO do we put the similarity matrix inside look_at?
        #self.validation_data_pairs = [[self.validation_data[:, 0], self.validation_data[:, 1]], # All pairs  
        #                        [self.validation_data[:, 0], self.validation_data[:, 2]], # TODO change to BehavioralAssembly
        #                        [self.validation_data[:, 1], self.validation_data[:, 2]]]

        super(Hebart2023Accuracy, self).__init__(
            identifier=None, version=1,
            ceiling_func=None,
            parent=None,
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        candidate.start_task(BrainModel.Task.odd_one_out, None) 
        validation_data = self.validation_data
        predicted_odd_one_outs = candidate.look_at(self._assembly.stimulus_set, number_of_trials=self._number_of_trials)
        raw_score = self._metric(predicted_odd_one_outs, self._assembly)
        ceiling = self.ceiling
        score = (raw_score - 1/3) / (ceiling - 1/3)
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score

    # TODO do we put the similarity matrix inside look_at?
    #def sub2ind(array_shape, rows, cols): 
    #    ind = np.ix_(rows, cols)
    #    return np.ravel_multi_index(ind, array_shape)
