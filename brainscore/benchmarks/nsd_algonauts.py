import numpy as np
import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.model_interface import BrainModel
from brainscore.metrics import Score
from brainscore.metrics.nsd_algonauts_correlation import RegressedCorrelationAlgonauts, linear_regression, pearsonr_correlation
from brainscore.metrics.transformations import standard_error_of_the_mean
from brainscore.benchmarks._neural_common import average_repetition
from brainscore.utils import LazyLoad
from tqdm import tqdm
import xarray as xr
import os

# Constants
VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 3

BIBTEX_NSD = """@article{allen2022massive,
  title={A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence},
  author={Allen, Emily J and St-Yves, Ghislain and Wu, Yihan and Breedlove, Jesse L and Prince, Jacob S and Dowdle, Logan T and Nau, Matthias and Caron, Brad and Pestilli, Franco and Charest, Ian and others},
  journal={Nature neuroscience},
  volume={25},
  number={1},
  pages={116--126},
  year={2022},
  publisher={Nature Publishing Group US New York}
}
"""

BIBTEX_Algonauts = """@article{gifford2023algonauts,
  title={The algonauts project 2023 challenge: How the human brain makes sense of natural scenes},
  author={Gifford, Alessandro T and Lahner, Benjamin and Saba-Sadiya, Sari and Vilas, Martina G and Lascelles, Alex and Oliva, Aude and Kay, Kendrick and Roig, Gemma and Cichy, Radoslaw M},
  journal={arXiv preprint arXiv:2301.03198},
  year={2023}
}
"""

SUBJECTS = ["subject1", "subject2", "subject3", "subject4", "subject5", "subject6", "subject7", "subject8"]


# Benchmark Base
class NSDNeuralBenchmark(BenchmarkBase):
    def __init__(self, identifier, assemblies, assemblies_repetition, similarity_metric, visual_degrees, number_of_trials, ceiler, **kwargs):
        """
        Initialize the NSDNeuralBenchmark.
        Args:
            identifier: Benchmark identifier.
            assemblies: List of assemblies.
            assemblies_repetition: Assemblies with repetitions.
            similarity_metric: Similarity metric.
            visual_degrees: Visual degrees.
            number_of_trials: Number of trials.
            ceiler: Ceiling function.
        """
        super(NSDNeuralBenchmark, self).__init__(identifier=identifier, **kwargs)
        self._assemblies = assemblies
        self.assemblies_repetition = assemblies_repetition
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
        subject_scores = []
        for i, subject_assembly in tqdm(enumerate(self._assemblies), desc='Computing score for all Subjects'):
            # record from model
            candidate.start_recording(self.region, time_bins=self.timebins)
            stimulus_set = place_on_screen(subject_assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                    source_visual_degrees=self._visual_degrees)
            source_assembly = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
            if 'time_bin' in source_assembly.dims:
                source_assembly = source_assembly.squeeze('time_bin')  # static case for these benchmarks
            source_assembly = source_assembly.transpose('presentation', 'neuroid')
            
            # compute score
            raw_score = self._similarity_metric(source_assembly, subject_assembly)
            if self.ceiling_func is None:
                subject_ceiling = subject_assembly['algonauts_noise_ceiling'].values
            else:
                subject_ceiling = self.ceiling_func(self.assemblies_repetition[i])
            
            normalized_score = ceil_score(raw_score, subject_ceiling).mean('dim_0', keep_attrs=True)
            normalized_score['subject'] = SUBJECTS[i]
            subject_scores.append(normalized_score)
        subject_scores_merged = xr.concat(subject_scores, dim='subject')
        subject_scores_merged.attrs['raw'] = np.concatenate([subj.attrs['raw'] for subj in subject_scores])
        subject_scores_merged.attrs['ceiling'] = np.concatenate([subj.attrs['ceiling'] for subj in subject_scores])
        subject_scores_merged.attrs['sub'] = np.concatenate([[f'subject{i + 1}'] * len(subj.attrs['raw']) for i, subj in enumerate(subject_scores)])
        del subject_scores
        score = subject_scores_merged.mean(dim='subject', keep_attrs=True)
        score.attrs['error'] = standard_error_of_the_mean(subject_scores_merged, 'subject')
        return score

# Benchmark Implementation
def _NSD2022Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    # Load assemblies
    assemblies = load_subject_assemblies(average_repetitions=True, region=region)
    if ceiler is None:
        assemblies_repetition = None
    else:
        assemblies_repetition = load_subject_assemblies(average_repetitions=True, region=region)
    return NSDNeuralBenchmark(identifier=f'NSD2022.{region}-{identifier_metric_suffix}', version=0, 
                           assemblies=assemblies, assemblies_repetition=assemblies_repetition, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=None,
                           ceiler=ceiler,
                           parent=region,
                           bibtex=BIBTEX_NSD)

def NSD2022WholeBrain():
    return _NSD2022Region(region='whole_brain', identifier_metric_suffix='algonauts-linear',
                           similarity_metric=RegressedCorrelationAlgonauts(
                               regression=linear_regression(), correlation=pearsonr_correlation()),
                           ceiler=None)

def NSD2022V1v():
    return _NSD2022Region(region='V1v', identifier_metric_suffix='algonauts-linear',
                                       similarity_metric=RegressedCorrelationAlgonauts(
                                           regression=linear_regression(), correlation=pearsonr_correlation()),
                                       ceiler=None)

def NSD2022V1d():
    return _NSD2022Region(region='V1d', identifier_metric_suffix='algonauts-linear',
                           similarity_metric=RegressedCorrelationAlgonauts(
                               regression=linear_regression(), correlation=pearsonr_correlation()),
                           ceiler=None)

def NSD2022V2v():
    return _NSD2022Region(region='V2v', identifier_metric_suffix='algonauts-linear',
                           similarity_metric=RegressedCorrelationAlgonauts(
                               regression=linear_regression(), correlation=pearsonr_correlation()),
                           ceiler=None)

def NSD2022V2d():
    return _NSD2022Region(region='V2d', identifier_metric_suffix='algonauts-linear',
                           similarity_metric=RegressedCorrelationAlgonauts(
                               regression=linear_regression(), correlation=pearsonr_correlation()),
                           ceiler=None)

def NSD2022V3v():
    return _NSD2022Region(region='V3d', identifier_metric_suffix='algonauts-linear',
                           similarity_metric=RegressedCorrelationAlgonauts(
                               regression=linear_regression(), correlation=pearsonr_correlation()),
                           ceiler=None)

def NSD2022V3d():
    return _NSD2022Region(region='V3d', identifier_metric_suffix='algonauts-linear',
                           similarity_metric=RegressedCorrelationAlgonauts(
                               regression=linear_regression(), correlation=pearsonr_correlation()),
                           ceiler=None)

def NSD2022V4():
    return _NSD2022Region(region='hV4', identifier_metric_suffix='algonauts-linear',
                           similarity_metric=RegressedCorrelationAlgonauts(
                               regression=linear_regression(), correlation=pearsonr_correlation()),
                           ceiler=None)

# Helper functions
def load_subject_assemblies(average_repetitions, region):
    assemblies = [] 
    for subject in tqdm(SUBJECTS, desc='Collecting assemblies for all subjects'):
        assemblies.append(LazyLoad(lambda subject=subject: load_subject_assembly(average_repetitions=average_repetitions, region=region, subject=subject)))
    return assemblies

def load_subject_assembly(average_repetitions, region, subject):
    assembly = brainscore.get_assembly(name=f'NSD2022_{subject}_assembly')
    region_mapping = {
        'V1v': 'hV1v',
        'V1d': 'hV1d',
        'V2v': 'hV2v',
        'V2d': 'hV2d',
        'V3v': 'hV3v',
        'V3d': 'hV3d',
        'hV4': 'hV4',
    }
    if region == 'whole_brain':
        region_ = 'nsd_general_rsc_most_responsive'
        assembly = assembly.sel(region_whole_brain=region_)
    elif region in region_mapping.keys():
        assembly = assembly.sel(region_prf_visualrois=region)
        region = region_mapping.get(region, region)  # Use the mapping or keep the original if not found

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
        timebins = [timebins]  # only a single time-bin
    return timebins

def ceil_score(raw_score, ceiling):
    # ignore voxels with zero self-consistency
    noise_ceiling_zeroidx = np.argwhere(ceiling == 0)
    ceiling = np.delete(ceiling, noise_ceiling_zeroidx)
    raw_score = np.delete(raw_score, noise_ceiling_zeroidx)
    
    # set nan to 0
    idx_nan = np.isnan(raw_score)
    if any(idx_nan):
        raw_score[idx_nan] = 0 
    
    # normalize with ceiling
    ceiled_score = (raw_score ** 2) / ceiling
    ceiled_score[ceiled_score > 1] = 1
    ceiled_score.attrs[Score.RAW_VALUES_KEY] = raw_score.values
    ceiled_score.attrs['ceiling'] = ceiling
