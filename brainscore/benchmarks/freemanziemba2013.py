import brainscore
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency, CKAConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.cka import CKACrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation, \
    single_regression
from brainscore.utils import LazyLoad
from result_caching import store

VISUAL_DEGREES = 4
NUMBER_OF_TRIALS = 20

BIBTEX = """﻿@Article{Freeman2013,
                author={Freeman, Jeremy
                and Ziemba, Corey M.
                and Heeger, David J.
                and Simoncelli, Eero P.
                and Movshon, J. Anthony},
                title={A functional and perceptual signature of the second visual area in primates},
                journal={Nature Neuroscience},
                year={2013},
                month={Jul},
                day={01},
                volume={16},
                number={7},
                pages={974-981},
                abstract={The authors examined neuronal responses in V1 and V2 to synthetic texture stimuli that replicate higher-order statistical dependencies found in natural images. V2, but not V1, responded differentially to these textures, in both macaque (single neurons) and human (fMRI). Human detection of naturalistic structure in the same images was predicted by V2 responses, suggesting a role for V2 in representing natural image structure.},
                issn={1546-1726},
                doi={10.1038/nn.3402},
                url={https://doi.org/10.1038/nn.3402}
                }
            """


def _MovshonFreemanZiemba2013Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(True, region=region))
    return NeuralBenchmark(identifier=f'movshon.FreemanZiemba2013.{region}-{identifier_metric_suffix}', version=2,
                           assembly=assembly, similarity_metric=similarity_metric, parent=region,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           bibtex=BIBTEX)


def MovshonFreemanZiemba2013V1PLS():
    return _MovshonFreemanZiemba2013Region('V1', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=pls_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=InternalConsistency())


def MovshonFreemanZiemba2013V1Single():
    return _MovshonFreemanZiemba2013Region('V1', identifier_metric_suffix='single',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=single_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=InternalConsistency())


def MovshonFreemanZiemba2013V1RDM():
    return _MovshonFreemanZiemba2013Region('V1', identifier_metric_suffix='rdm',
                                           similarity_metric=RDMCrossValidated(
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=RDMConsistency())


def MovshonFreemanZiemba2013V1CKA():
    return _MovshonFreemanZiemba2013Region('V1', identifier_metric_suffix='cka',
                                           similarity_metric=CKACrossValidated(
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=CKAConsistency())


def MovshonFreemanZiemba2013V2PLS():
    return _MovshonFreemanZiemba2013Region('V2', identifier_metric_suffix='pls',
                                           similarity_metric=CrossRegressedCorrelation(
                                               regression=pls_regression(), correlation=pearsonr_correlation(),
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=InternalConsistency())


def MovshonFreemanZiemba2013V2RDM():
    return _MovshonFreemanZiemba2013Region('V2', identifier_metric_suffix='rdm',
                                           similarity_metric=RDMCrossValidated(
                                               crossvalidation_kwargs=dict(stratification_coord='texture_type')),
                                           ceiler=RDMConsistency())


@store()
def load_assembly(average_repetitions, region, access='private'):
    assembly = brainscore.get_assembly(f'movshon.FreemanZiemba2013.{access}')
    assembly = assembly.sel(region=region)
    assembly = assembly.stack(neuroid=['neuroid_id'])  # work around xarray multiindex issues
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly.load()
    time_window = (50, 200)
    assembly = assembly.sel(time_bin=[(t, t + 1) for t in range(*time_window)])
    assembly = assembly.mean(dim='time_bin', keep_attrs=True)
    assembly = assembly.expand_dims('time_bin_start').expand_dims('time_bin_end')
    assembly['time_bin_start'], assembly['time_bin_end'] = [time_window[0]], [time_window[1]]
    assembly = assembly.stack(time_bin=['time_bin_start', 'time_bin_end'])
    assembly = assembly.squeeze('time_bin')
    assembly = assembly.transpose('presentation', 'neuroid')
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly
