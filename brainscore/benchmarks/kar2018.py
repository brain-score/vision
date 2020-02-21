import brainscore
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.utils import LazyLoad


def DicarloKar2018hvmPLS():
    assembly_repetition = LazyLoad(lambda: load_assembly(stimuli='hvm', average_repetitions=False))
    assembly = LazyLoad(lambda: load_assembly(stimuli='hvm', average_repetitions=True))
    similarity_metric = CrossRegressedCorrelation(
        regression=pls_regression(), correlation=pearsonr_correlation(),
        crossvalidation_kwargs=dict(stratification_coord='object_name'))
    ceiler = InternalConsistency()
    return NeuralBenchmark(identifier=f'dicarlo.Kar2018hvm-pls', version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent='IT', paper_link=None)


def DicarloKar2018cocoPLS():
    assembly_repetition = LazyLoad(lambda: load_assembly(stimuli='cocogray', average_repetitions=False))
    assembly = LazyLoad(lambda: load_assembly(stimuli='cocogray', average_repetitions=True))
    similarity_metric = CrossRegressedCorrelation(
        regression=pls_regression(), correlation=pearsonr_correlation(),
        crossvalidation_kwargs=dict(stratification_coord='label'))
    ceiler = InternalConsistency()
    return NeuralBenchmark(identifier=f'dicarlo.Kar2018coco-pls', version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent='IT', paper_link=None)


def load_assembly(stimuli, average_repetitions):
    assembly = brainscore.get_assembly(name=f'dicarlo.Kar2018{stimuli}')
    assembly = assembly.squeeze("time_bin")
    assembly.load()
    assembly = assembly.transpose('presentation', 'neuroid')
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly
