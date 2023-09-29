import numpy as np
import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.model_interface import BrainModel
from brainscore.metrics import Score
from brainscore.metrics.nsd_algonauts_correlation import RegressedCorrelationAlgonauts, linear_regression, pearsonr_correlation
from brainscore.metrics.ceiling import NSDCeiling
from brainscore.benchmarks._neural_common import average_repetition
from brainscore.utils import LazyLoad
from tqdm import tqdm 

VISUAL_DEGREES = 8.4

NUMBER_OF_TRIALS = 3

BIBTEX = """@article{allen2022massive,
  title={A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence},
  author={Allen, Emily J and St-Yves, Ghislain and Wu, Yihan and Breedlove, Jesse L and Prince, Jacob S and Dowdle, Logan T and Nau, Matthias and Caron, Brad and Pestilli, Franco and Charest, Ian and others},
  journal={Nature neuroscience},
  volume={25},
  number={1},
  pages={116--126},
  year={2022},
  publisher={Nature Publishing Group US New York}
}

@article{gifford2023algonauts,
  title={The algonauts project 2023 challenge: How the human brain makes sense of natural scenes},
  author={Gifford, Alessandro T and Lahner, Benjamin and Saba-Sadiya, Sari and Vilas, Martina G and Lascelles, Alex and Oliva, Aude and Kay, Kendrick and Roig, Gemma and Cichy, Radoslaw M},
  journal={arXiv preprint arXiv:2301.03198},
  year={2023}
}
"""

SUBJECTS = ["subject1", "subject2", "subject3", "subject4", "subject5", "subject6", "subject7", "subject8"]


### BENCHMARK BASE #############################################################################################################

class NSDNeuralBenchmark(BenchmarkBase):
    def __init__(self, identifier, assemblies, similarity_metric, visual_degrees, number_of_trials, ceiler, **kwargs):
        super(NSDNeuralBenchmark, self).__init__(identifier=identifier, **kwargs)
        self._assemblies = assemblies
        self._similarity_metric = similarity_metric
        region = np.unique(self._assemblies[0]['region'])
        assert len(region) == 1
        self.region = region[0]
        timebins = timebins_from_assembly(self._assemblies[0])
        self.timebins = timebins
        self._visual_degrees = visual_degrees
        self._number_of_trials = number_of_trials
        self.ceiling_func = ceiler

    def __call__(self, candidate: BrainModel):
        final_scores = []
        for i, subject in tqdm(enumerate(SUBJECTS), desc='Computing raw score for all Subjects', total=len(SUBJECTS)):
            print('start recording ANN responses')
            candidate.start_recording(self.region, time_bins=self.timebins)
            stimulus_set = place_on_screen(self._assemblies[i].stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                    source_visual_degrees=self._visual_degrees)
            source_assembly = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
            print('source_assembly extracted')
            if 'time_bin' in source_assembly.dims:
                source_assembly = source_assembly.squeeze('time_bin')  # static case for these benchmarks
            source_assembly = source_assembly.transpose('presentation', 'neuroid')
            print('computing the raw score')
            raw_score = self._similarity_metric(source_assembly, self._assemblies[i])
            ceil = self.ceiling_func(self._assemblies[i])
            
            normalized_score = normalize_with_NC(raw_score, ceil)

            final_scores = np.concatenate((final_scores, normalized_score))

        score_center = np.mean(final_scores)
        score_error = np.std(final_scores) / np.sqrt(len(final_scores))

        score =  Score([score_center, score_error],
                            coords={**{'aggregation': ['center', 'error']}},
                            dims=('aggregation',) )
        return score


### BENCHMARK IMPLEMENTATION ##################################################################################################################

def _NSD2023Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assemblies = [] 
    for subject in tqdm(SUBJECTS, desc='Collecting assemblies for all subjects'):
        assembly_repetition = LazyLoad(lambda subject=subject: load_assembly(average_repetitions=False, subject=subject))
        assemblies.append(LazyLoad(lambda region=region, subject=subject: load_assembly(average_repetitions=True, region=region, subject=subject)))
    return NSDNeuralBenchmark(identifier=f'bocini-nsd-2023.{region}-{identifier_metric_suffix}', version=0, 
                           assemblies=assemblies, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func = None,
                           ceiler = ceiler,
                           parent=region,
                           bibtex=BIBTEX)

def NSD2023WholeBrain():
    return _NSD2023Region(region='whole_brain', identifier_metric_suffix='algonauts-metric',
                                       similarity_metric=RegressedCorrelationAlgonauts(
                                           regression=linear_regression(), correlation=pearsonr_correlation()),
                                       ceiler=NSDCeiling())

def NSD2023V1v():
    return _NSD2023Region(region='V1v', identifier_metric_suffix='algonauts-metric',
                                       similarity_metric=RegressedCorrelationAlgonauts(
                                           regression=linear_regression(), correlation=pearsonr_correlation()),
                                       ceiler=NSDCeiling())

def NSD2023V2v():
    return _NSD2023Region(region='V2v', identifier_metric_suffix='algonauts-metric',
                                       similarity_metric=RegressedCorrelationAlgonauts(
                                           regression=linear_regression(), correlation=pearsonr_correlation()),
                                       ceiler=NSDCeiling())

def NSD2023V4():
    return _NSD2023Region(region='hV4', identifier_metric_suffix='algonauts-metric',
                                       similarity_metric=RegressedCorrelationAlgonauts(
                                           regression=linear_regression(), correlation=pearsonr_correlation()),
                                       ceiler=NSDCeiling())



### Helper functions ########################################################################################################################

def load_assembly(average_repetitions, region, subject ):
    assembly = brainscore.get_assembly(name=f'bocini-nsd-2023_{subject}_assembly')
    region_mapping = {
        'V1v': 'hV1v',
        'V2v': 'hV2v',
        'hV4': 'hV4',
    }
    if region == 'whole_brain':
        region_ = 'nsd_general_rsc_most_responsive'
        assembly = assembly.sel(region_whole_brain=region_)
    elif region in region_mapping.keys():
        assembly = assembly.sel(region_prf_visualrois=region)
        region = region_mapping.get(region, region) # Use the mapping or keep the original if not found

    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly = assembly.squeeze("time_bin")
    assembly.load()
    assembly = assembly.transpose('presentation', 'neuroid')
    assert NUMBER_OF_TRIALS == len(np.unique(assembly.coords['repetition']))
    if average_repetitions:
        avg_assembly = average_repetition(assembly.reset_index(("stim_number"), drop=True))
        assembly = avg_assembly.sortby(avg_assembly.counts_id)
    return assembly

def timebins_from_assembly(assembly):
    timebins = assembly['time_bin'].values
    if 'time_bin' not in assembly.dims:
        timebins = [timebins]  # only single time-bin
    return timebins

def normalize_with_NC(raw_score, ceil):
    noise_ceiling_zeroidx = np.argwhere(ceil == 0)
    ceil =  np.delete(ceil, noise_ceiling_zeroidx)

    raw_score = raw_score.sel(aggregation='center').values
    raw_score = raw_score.item()
    raw_score[raw_score<0] = 0
    raw_score = np.delete(raw_score, noise_ceiling_zeroidx)
    idx_nan = np.isnan(raw_score)
    raw_score[idx_nan] = 0 
    
    normalized_score = np.divide(raw_score**2, ceil) * 100
    normalized_score[normalized_score>100]=100

    return normalized_score

