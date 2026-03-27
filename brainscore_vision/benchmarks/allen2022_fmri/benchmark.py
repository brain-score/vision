import pandas as pd
import xarray as xr

from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision import load_dataset, load_metric
from brainscore_vision.benchmark_helpers.neural_common import (
    TrainTestNeuralBenchmark, RSABenchmark, average_repetition,
    filter_reliable_neuroids,
)
from brainscore_vision.metrics.regression_correlation.metric import ALPHA_LIST
from brainscore_vision.utils import LazyLoad

BIBTEX = """@article{allen_massive_2022,
    title = {A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence},
    volume = {25},
    issn = {1097-6256},
    doi = {10.1038/s41593-021-00962-x},
    journal = {Nature Neuroscience},
    author = {Allen, Emily J. and St-Yves, Ghislain and Wu, Yihan and Breedlove, Jesse L.
              and Prince, Jacob S. and Dowdle, Logan T. and Nau, Matthias and Caron, Brad
              and Pestilli, Franco and Charest, Ian and Hutchinson, J. Benjamin
              and Naselaris, Thomas and Kay, Kendrick},
    year = {2022},
    pages = {116--126},
}"""

# NSD: 8.4 x 8.4 deg square aperture (Allen et al. 2022, Methods)
VISUAL_DEGREES = 8.4

# Noise ceiling threshold in percentage units (nc_testset stored as 0-100).
# Voxels with nc_testset <= this value are dropped by filter_reliable_neuroids.
NOISE_CEILING_THRESHOLD = 0.3 * 100

# ROI definitions:
# V1, V2, V4: pRF-based ROIs from NSD (lh/rh.visualrois.nii.gz).
# IT: NSD "streams" ventral parcellation (lh/rh.streams.nii.gz, label=5).
#   This broad ventral-stream definition follows the Algonauts 2023 challenge
#   (Gifford et al., arXiv:2301.03198) which used the same NSD streams ROI.
#   See data/allen2022_fmri/data_packaging/notebooks/06_scientific_validation.ipynb
#   for comparison with the Glasser HCP-MMP1 9-parcel IT definition (Hebart2023).

def _Allen2022fmri(region,
                   similarity_metric,
                   identifier_metric_suffix,
                   dataset_prefix='Allen2022_fmri',
                   alpha_coord=None,
                   per_voxel_ceilings=False,
                   visual_degrees=VISUAL_DEGREES,
                   ceiler=load_metric('internal_consistency'),
                   noise_ceiling_threshold=NOISE_CEILING_THRESHOLD):
    number_of_trials = 1
    train_assembly = LazyLoad(lambda region=region, nct=noise_ceiling_threshold, dp=dataset_prefix:
                              load_assembly(region=region,
                                            split='train',
                                            average_repetitions=False,
                                            dataset_prefix=dp,
                                            noise_ceiling_threshold=nct))
    test_assembly = LazyLoad(lambda region=region, nct=noise_ceiling_threshold, dp=dataset_prefix:
                             load_assembly(region=region,
                                           split='test',
                                           average_repetitions=True,
                                           dataset_prefix=dp,
                                           noise_ceiling_threshold=nct))
    test_assembly_repetition = LazyLoad(lambda region=region, nct=noise_ceiling_threshold, dp=dataset_prefix:
                                        load_assembly(region=region,
                                                      split='test',
                                                      average_repetitions=False,
                                                      dataset_prefix=dp,
                                                      noise_ceiling_threshold=nct))
    return TrainTestNeuralBenchmark(
        identifier=f'{dataset_prefix}.{region}-{identifier_metric_suffix}',
        version=1,
        ceiling_func=lambda: ceiler(test_assembly_repetition),
        train_assembly=train_assembly,
        test_assembly=test_assembly,
        similarity_metric=similarity_metric,
        alpha_coord=alpha_coord,
        per_voxel_ceilings=per_voxel_ceilings,
        visual_degrees=visual_degrees,
        number_of_trials=number_of_trials,
        parent=region,
        bibtex=BIBTEX)


def Allen2022fmri(region: str, metric_type: str,
                  dataset_prefix: str = 'Allen2022_fmri',
                  alphas: list = ALPHA_LIST):
    similarity_metric = load_metric(f'{metric_type}_split', alphas=alphas)
    return _Allen2022fmri(region, similarity_metric=similarity_metric,
                          identifier_metric_suffix=metric_type,
                          dataset_prefix=dataset_prefix,
                          alpha_coord='subject', per_voxel_ceilings=False)


def Allen2022fmriRSA(region: str,
                     dataset_prefix: str = 'Allen2022_fmri') -> RSABenchmark:
    assembly = LazyLoad(lambda region=region, dp=dataset_prefix:
                        load_full_assembly(region=region, dataset_prefix=dp))
    return RSABenchmark(
        identifier=f'{dataset_prefix}.{region}-rdm',
        version=1,
        assembly=assembly,
        region=region,
        visual_degrees=VISUAL_DEGREES,
        number_of_trials=1,
        bibtex=BIBTEX,
    )


def load_assembly(region, split, average_repetitions,
                  dataset_prefix='Allen2022_fmri',
                  noise_ceiling_threshold=NOISE_CEILING_THRESHOLD):
    assembly = load_dataset(f'{dataset_prefix}_{split}')
    assembly = filter_reliable_neuroids(assembly, noise_ceiling_threshold, 'nc_testset')
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly.load()
    assembly = assembly.isel(time_bin=0)
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly


def load_full_assembly(region: str,
                       dataset_prefix: str = 'Allen2022_fmri',
                       noise_ceiling_threshold: float = NOISE_CEILING_THRESHOLD):
    """Load all images (train + test), repetitions averaged. For RSA benchmarks."""
    train = load_assembly(region, split='train', average_repetitions=True,
                          dataset_prefix=dataset_prefix,
                          noise_ceiling_threshold=noise_ceiling_threshold)
    test = load_assembly(region, split='test', average_repetitions=True,
                         dataset_prefix=dataset_prefix,
                         noise_ceiling_threshold=noise_ceiling_threshold)

    combined = type(train)(xr.concat([train, test], dim='presentation'))

    train_stimuli = train.stimulus_set
    test_stimuli = test.stimulus_set
    combined_stimuli = StimulusSet(pd.concat([train_stimuli, test_stimuli], ignore_index=True))
    combined_stimuli.stimulus_paths = {**train_stimuli.stimulus_paths, **test_stimuli.stimulus_paths}
    combined_stimuli.identifier = f'{dataset_prefix}_full'
    combined.attrs['stimulus_set'] = combined_stimuli

    return combined
