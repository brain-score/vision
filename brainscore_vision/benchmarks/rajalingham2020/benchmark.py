import numpy as np

import brainscore_vision
from brainscore_vision.benchmark_helpers._neural_common import NeuralBenchmark, average_repetition
from brainscore_vision.metrics.ceiling import InternalConsistency
from brainscore_vision.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore_vision.utils import LazyLoad


VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 33
BIBTEX = """@article {Rajalingham2020,
            author = {Rajalingham, Rishi and Kar, Kohitij and Sanghavi, Sachi and Dehaene, Stanislas and DiCarlo, James J.},
            title = {The inferior temporal cortex is a potential cortical precursor of orthographic processing in untrained monkeys},
            journal = {Nature Communications},
            year = {2020},
            month = {Aug},
            day = {04},
            volume = {11},
            number = {1},
            pages = {3886},
            abstract = {The ability to recognize written letter strings is foundational to human reading, but the underlying neuronal mechanisms remain largely unknown. Recent behavioral research in baboons suggests that non-human primates may provide an opportunity to investigate this question. We recorded the activity of hundreds of neurons in V4 and the inferior temporal cortex (IT) while na{\"i}ve macaque monkeys passively viewed images of letters, English words and non-word strings, and tested the capacity of those neuronal representations to support a battery of orthographic processing tasks. We found that simple linear read-outs of IT (but not V4) population responses achieved high performance on all tested tasks, even matching the performance and error patterns of baboons on word classification. These results show that the IT cortex of untrained primates can serve as a precursor of orthographic processing, suggesting that the acquisition of reading in humans relies on the recycling of a brain network evolved for other visual functions.},
            issn = {2041-1723},
            doi = {10.1038/s41467-020-17714-3},
            url = {https://doi.org/10.1038/s41467-020-17714-3}}"""


def _DicarloRajalingham2020Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region))
    return NeuralBenchmark(identifier=f'dicarlo.Rajalingham2020.{region}-{identifier_metric_suffix}', version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)


def DicarloRajalingham2020ITPLS():
    return _DicarloRajalingham2020Region('IT', identifier_metric_suffix='pls',
                                         similarity_metric=CrossRegressedCorrelation(
                                             regression=pls_regression(), correlation=pearsonr_correlation(),
                                             crossvalidation_kwargs=dict(stratification_coord=None)),
                                         ceiler=InternalConsistency())


def load_assembly(average_repetitions, region):
    assembly = brainscore_vision.load_dataset(identifier=f'dicarlo.Rajalingham2020')
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly.load()
    assembly = assembly.squeeze('time_bin')
    assert NUMBER_OF_TRIALS == len(np.unique(assembly.coords['repetition']))
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly
