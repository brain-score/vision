from brainscore.assemblies.private import assembly_loaders
from brainscore.benchmarks._neural_common import NeuralBenchmark
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, pls_regression, \
    pearsonr_correlation


def ToliasCadena2017PLS():
    loader = assembly_loaders[f'tolias.Cadena2017']
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation(
        regression=pls_regression(),
        correlation=pearsonr_correlation(),
        crossvalidation_kwargs={'stratification_coord': None})
    identifier = f'tolias.Cadena2017-pls'
    ceiler = InternalConsistency(split_coord='repetition_id')

    def ceiling():
        # This assembly has many stimuli that are only shown to a subset of the neurons.
        # When the loader runs with `average_repetition=True`, it automatically drops nan cells,
        # but for the `assembly_repetition`, it keeps all the rows.
        # If we now use the usual ceiling approach, the two halves will end up with NaN values
        # due to stimuli not being shown to neurons, which doesn't let us correlate.
        # Instead, we here drop all NaN cells and their corresponding stimuli,
        # which keeps only 43% of the original presentation rows, but lets us correlate again.
        assembly_nonan, stimuli = loader.dropna(assembly_repetition, assembly_repetition.attrs['stimulus_set'])
        return ceiler(assembly_nonan)

    return NeuralBenchmark(identifier=identifier, assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=ceiling)


def ToliasCadena2017Mask():
    loader = assembly_loaders[f'tolias.Cadena2017']
    assembly_repetition = loader(average_repetition=False)
    assembly = loader(average_repetition=True)
    assembly.stimulus_set.name = assembly.stimulus_set_name

    similarity_metric = CrossRegressedCorrelation(
        regression=mask_regression(),
        correlation=pearsonr_correlation(),
        crossvalidation_kwargs={'splits': 4, 'stratification_coord': None})
    identifier = f'tolias.Cadena2017-mask'
    ceiler = InternalConsistency(split_coord='repetition_id')
    return NeuralBenchmark(identifier=identifier, assembly=assembly, similarity_metric=similarity_metric,
                           ceiling_func=lambda: ceiler(assembly_repetition))
