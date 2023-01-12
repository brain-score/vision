import functools

import brainscore_vision

from brainscore_vision.benchmark_helpers._neural_common import NeuralBenchmark, average_repetition
from brainscore_vision.metrics.ceiling import InternalConsistency, RDMConsistency
from brainscore_vision.metrics.rdm import RDMCrossValidated
from brainscore_vision.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation

VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 50
BIBTEX = """@article {Majaj13402,
            author = {Majaj, Najib J. and Hong, Ha and Solomon, Ethan A. and DiCarlo, James J.},
            title = {Simple Learned Weighted Sums of Inferior Temporal Neuronal Firing Rates Accurately Predict Human Core Object Recognition Performance},
            volume = {35},
            number = {39},
            pages = {13402--13418},
            year = {2015},
            doi = {10.1523/JNEUROSCI.5181-14.2015},
            publisher = {Society for Neuroscience},
            abstract = {To go beyond qualitative models of the biological substrate of object recognition, we ask: can a single ventral stream neuronal linking hypothesis quantitatively account for core object recognition performance over a broad range of tasks? We measured human performance in 64 object recognition tests using thousands of challenging images that explore shape similarity and identity preserving object variation. We then used multielectrode arrays to measure neuronal population responses to those same images in visual areas V4 and inferior temporal (IT) cortex of monkeys and simulated V1 population responses. We tested leading candidate linking hypotheses and control hypotheses, each postulating how ventral stream neuronal responses underlie object recognition behavior. Specifically, for each hypothesis, we computed the predicted performance on the 64 tests and compared it with the measured pattern of human performance. All tested hypotheses based on low- and mid-level visually evoked activity (pixels, V1, and V4) were very poor predictors of the human behavioral pattern. However, simple learned weighted sums of distributed average IT firing rates exactly predicted the behavioral pattern. More elaborate linking hypotheses relying on IT trial-by-trial correlational structure, finer IT temporal codes, or ones that strictly respect the known spatial substructures of IT ({\textquotedblleft}face patches{\textquotedblright}) did not improve predictive power. Although these results do not reject those more elaborate hypotheses, they suggest a simple, sufficient quantitative model: each object recognition task is learned from the spatially distributed mean firing rates (100 ms) of \~{}60,000 IT neurons and is executed as a simple weighted sum of those firing rates.SIGNIFICANCE STATEMENT We sought to go beyond qualitative models of visual object recognition and determine whether a single neuronal linking hypothesis can quantitatively account for core object recognition behavior. To achieve this, we designed a database of images for evaluating object recognition performance. We used multielectrode arrays to characterize hundreds of neurons in the visual ventral stream of nonhuman primates and measured the object recognition performance of \&gt;100 human observers. Remarkably, we found that simple learned weighted sums of firing rates of neurons in monkey inferior temporal (IT) cortex accurately predicted human performance. Although previous work led us to expect that IT would outperform V4, we were surprised by the quantitative precision with which simple IT-based linking hypotheses accounted for human behavior.},
            issn = {0270-6474},
            URL = {https://www.jneurosci.org/content/35/39/13402},
            eprint = {https://www.jneurosci.org/content/35/39/13402.full.pdf},
            journal = {Journal of Neuroscience}}"""


def _DicarloMajajHong2015Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = lambda region=region: load_assembly(average_repetitions=False, region=region)
    assembly = lambda region=region: load_assembly(average_repetitions=True, region=region)
    return NeuralBenchmark(identifier=f'dicarlo.MajajHong2015.{region}-{identifier_metric_suffix}', version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)


def DicarloMajajHong2015V4PLS():
    return _DicarloMajajHong2015Region('V4', identifier_metric_suffix='pls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=pls_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015ITPLS():
    return _DicarloMajajHong2015Region('IT', identifier_metric_suffix='pls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=pls_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015V4Mask():
    return _DicarloMajajHong2015Region('V4', identifier_metric_suffix='mask',
                                       similarity_metric=ScaledCrossRegressedCorrelation(
                                           regression=mask_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015ITMask():
    return _DicarloMajajHong2015Region('IT', identifier_metric_suffix='mask',
                                       similarity_metric=ScaledCrossRegressedCorrelation(
                                           regression=mask_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015V4RDM():
    return _DicarloMajajHong2015Region('V4', identifier_metric_suffix='rdm',
                                       similarity_metric=RDMCrossValidated(
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=RDMConsistency())


def DicarloMajajHong2015ITRDM():
    return _DicarloMajajHong2015Region('IT', identifier_metric_suffix='rdm',
                                       similarity_metric=RDMCrossValidated(
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=RDMConsistency())


def MajajHongV4PublicBenchmark():
    return _standard_benchmark('dicarlo.MajajHong2015.V4.public',
                               load_assembly=functools.partial(load_assembly, region='V4', access='public'),
                               visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                               stratification_coord='object_name', bibtex=BIBTEX)


def MajajHongITPublicBenchmark():
    return _standard_benchmark('dicarlo.MajajHong2015.IT.public',
                               load_assembly=functools.partial(load_assembly, region='IT', access='public'),
                               visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                               stratification_coord='object_name', bibtex=BIBTEX)


def load_assembly(average_repetitions, region, access='private'):
    assembly = brainscore_vision.load_dataset(f'dicarlo.MajajHong2015.{access}')
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly = assembly.squeeze("time_bin")
    assembly.load()
    assembly = assembly.transpose('presentation', 'neuroid')
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly


def _standard_benchmark(identifier, load_assembly, visual_degrees, number_of_trials, stratification_coord, bibtex):
    assembly_repetition = lambda: load_assembly(average_repetitions=False)
    assembly = lambda: load_assembly(average_repetitions=True)
    similarity_metric = CrossRegressedCorrelation(
        regression=pls_regression(), correlation=pearsonr_correlation(),
        crossvalidation_kwargs=dict(stratification_coord=stratification_coord))
    ceiler = InternalConsistency()
    return NeuralBenchmark(identifier=f"{identifier}-pls", version=1,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=visual_degrees, number_of_trials=number_of_trials,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=None,
                           bibtex=bibtex)

