import brainscore
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.benchmarks._neural_common_extra import NeuralBenchmarkCovariate, ToleranceCeiling, NeuralBenchmarkImageDir, \
    NeuralBenchmarkCovariateGram, CacheFeaturesCovariate
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency, ToleranceConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation
from brainscore.metrics.regression_extra import CrossRegressedCorrelationCovariate, semipartial_regression, semipartial_pls, \
    ToleranceCrossValidation, CrossRegressedCorrelationDrew, old_gram_control_regression, old_gram_control_pls
from brainscore.utils import LazyLoad

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


def load_assembly(average_repetitions,region, name='dicarlo.MajajHong2015'):
    assembly = brainscore.get_assembly(name=name)
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly = assembly.squeeze("time_bin")
    assembly.load()
    assembly = assembly.transpose('presentation', 'neuroid')
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly


def _DicarloMajajHong2015Region_lmh(region, identifier_metric_suffix, similarity_metric, ceiler, benchmark_identifier='dicarlo.MajajHong2015'):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region))
    return NeuralBenchmark(identifier=benchmark_identifier, version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)

def _DicarloMajajHong2015Region_lmh_imagedir(image_dir, region, identifier_metric_suffix, similarity_metric, ceiler, benchmark_identifier='dicarlo.MajajHong2015'):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region))
    return NeuralBenchmarkImageDir(identifier=benchmark_identifier, version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           image_dir=image_dir,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)

def _DicarloMajajHong2015Region_lmh_toleranceceiling(region, identifier_metric_suffix, similarity_metric, benchmark_identifier='dicarlo.MajajHong2015'):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    return ToleranceCeiling(identifier=benchmark_identifier, version=3,
                           assembly=assembly_repetition, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           parent=region,
                           bibtex=BIBTEX)

# def _DicarloMajajHong2015Region_lmh_masked(region, identifier_metric_suffix, similarity_metric, ceiler, benchmark_identifier='dicarlo.MajajHong2015'):
#     assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region, masked=True))
#     assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region, masked=True))
#     return NeuralBenchmark(identifier=benchmark_identifier, version=3,
#                            assembly=assembly, similarity_metric=similarity_metric,
#                            visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
#                            ceiling_func=lambda: ceiler(assembly_repetition),
#                            parent=region,
#                            bibtex=BIBTEX)

def _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir, region, identifier_metric_suffix, similarity_metric, ceiler, assembly_name='dicarlo.MajajHong2015'):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region, name=assembly_name))
    return NeuralBenchmarkCovariate(identifier=f'{assembly_name}.{region}-{identifier_metric_suffix}', version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           covariate_image_dir=covariate_image_dir,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)

def _DicarloMajajHong2015Region_lmh_covariate_cache_features(covariate_image_dir, region, identifier_metric_suffix, similarity_metric, ceiler, assembly_name='dicarlo.MajajHong2015'):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region, name=assembly_name))
    return CacheFeaturesCovariate(identifier=f'{assembly_name}.{region}-{identifier_metric_suffix}', version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           covariate_image_dir=covariate_image_dir,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)

def _DicarloMajajHong2015Region_lmh_covariate_gram(covariate_image_dir, gram, region, identifier_metric_suffix, similarity_metric, ceiler, benchmark_identifier='dicarlo.MajajHong2015'):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region, name=benchmark_identifier))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region))
    return NeuralBenchmarkCovariateGram(identifier=benchmark_identifier, version=3,
                                    assembly=assembly, similarity_metric=similarity_metric,
                                    covariate_image_dir=covariate_image_dir,
                                    visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                                    ceiling_func=lambda: ceiler(assembly_repetition),
                                    parent=region,
                                    bibtex=BIBTEX,
                                    gram=gram)

def DicarloMajajHong2015V4PLS_lmh():
    return _DicarloMajajHong2015Region_lmh('V4', identifier_metric_suffix='pls',
                                                similarity_metric=CrossRegressedCorrelation(
                                                    regression=pls_regression(), correlation=pearsonr_correlation(),
                                                    crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                train_size=0.5)),
                                                ceiler=InternalConsistency())


def DicarloMajajHong2015ITPLS_lmh():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='pls',
                                                similarity_metric=CrossRegressedCorrelation(
                                                    regression=pls_regression(), correlation=pearsonr_correlation(),
                                                    crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                train_size=0.5)),
                                                ceiler=InternalConsistency())



def DicarloMajajHong2015ITPLS_lmh_ty_01_neg():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='pls',
                                                similarity_metric=CrossRegressedCorrelation(
                                                    regression=pls_regression(), correlation=pearsonr_correlation(),
                                                    crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                csv_file='__majajhonglocal_halves_ty_0.1_neg.csv',
                                                                                parent_folder='./data/splits/')),
                                                ceiler=InternalConsistency())


def DicarloMajajHong2015ITPLS_lmh_ty_01_pos():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='pls',
                                                similarity_metric=CrossRegressedCorrelation(
                                                    regression=pls_regression(), correlation=pearsonr_correlation(),
                                                    crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                csv_file='__majajhonglocal_halves_ty_0.1_pos.csv',
                                                                                parent_folder='./data/splits/')),
                                                ceiler=InternalConsistency())


def DicarloMajajHong2015ITPLS_lmh_tz_01_neg():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='pls',
                                                similarity_metric=CrossRegressedCorrelation(
                                                    regression=pls_regression(), correlation=pearsonr_correlation(),
                                                    crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                csv_file='__majajhonglocal_halves_tz_0.1_neg.csv',
                                                                                parent_folder='./data/splits/')),
                                                ceiler=InternalConsistency())


def DicarloMajajHong2015ITPLS_lmh_tz_01_pos():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='pls',
                                                similarity_metric=CrossRegressedCorrelation(
                                                    regression=pls_regression(), correlation=pearsonr_correlation(),
                                                    crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                csv_file='__majajhonglocal_halves_tz_0.1_pos.csv',
                                                                                parent_folder='./data/splits/')),
                                                ceiler=InternalConsistency())

#######
# SPR
########

def DicarloMajajHong2015V4SPR_c_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='V4', identifier_metric_suffix='covr_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control =True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                          train_size=0.5)),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPR_c_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control =True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                          train_size=0.5)),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITSPR_c_lmh_ty_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITSPR_c_lmh_ty_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPR_c_lmh_tz_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPR_c_lmh_tz_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPR_c_lmh_s_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPR_c_lmh_s_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015V4SPR_nc_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='V4', identifier_metric_suffix='covr_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                          train_size=0.5)),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPR_nc_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                          train_size=0.5)),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITSPR_nc_lmh_ty_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITSPR_nc_lmh_ty_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPR_nc_lmh_tz_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPR_nc_lmh_tz_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPR_nc_lmh_s_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPR_nc_lmh_s_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covr_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_regression(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

#########
# SPPLS
#########

def DicarloMajajHong2015V4SPPLS_c_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='V4', identifier_metric_suffix='covpls_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control =True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                          train_size=0.5)),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPPLS_c_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control =True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                          train_size=0.5)),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITSPPLS_c_lmh_ty_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITSPPLS_c_lmh_ty_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPPLS_c_lmh_tz_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPPLS_c_lmh_tz_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPPLS_c_lmh_s_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPPLS_c_lmh_s_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=True),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015V4SPPLS_nc_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='V4', identifier_metric_suffix='covpls_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                          train_size=0.5)),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPPLS_nc_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                          train_size=0.5)),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITSPPLS_nc_lmh_ty_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITSPPLS_nc_lmh_ty_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPPLS_nc_lmh_tz_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPPLS_nc_lmh_tz_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_nocontrol',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPPLS_nc_lmh_s_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITSPPLS_nc_lmh_s_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='covpls_control',
                                                          similarity_metric=CrossRegressedCorrelationCovariate(
                                                              regression=semipartial_pls(covariate_control=False),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

#########
# Drew
#########

def DicarloMajajHong2015V4Drew_c_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='V4', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=True,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  train_size=0.5)),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITDrew_c_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=True,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  train_size=0.5)),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITDrew_c_lmh_ty_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=True,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITDrew_c_lmh_ty_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=True,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITDrew_c_lmh_tz_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=True,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITDrew_c_lmh_tz_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=True,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITDrew_c_lmh_s_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=True,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITDrew_c_lmh_s_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=True,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015V4Drew_nc_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='V4', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=False,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                          train_size=0.5)),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITDrew_nc_lmh():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=False,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                                          train_size=0.5)),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITDrew_nc_lmh_ty_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=False,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITDrew_nc_lmh_ty_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=False,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_ty_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITDrew_nc_lmh_tz_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=False,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITDrew_nc_lmh_tz_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=False,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_tz_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())

def DicarloMajajHong2015ITDrew_nc_lmh_s_01_pos():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=False,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_pos.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())


def DicarloMajajHong2015ITDrew_nc_lmh_s_01_neg():
    return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir='image_dicarlo_hvm_bg',
                                                          region='IT', identifier_metric_suffix='Drew',
                                                          similarity_metric=CrossRegressedCorrelationDrew(
                                                              covariate_control=False,
                                                              regression=pls_regression(),
                                                              correlation=pearsonr_correlation(),
                                                              crossvalidation_kwargs=dict(
                                                                  stratification_coord='object_name',
                                                                  csv_file='__majajhonglocal_halves_s_0.1_neg.csv',
                                                                  parent_folder='./data/splits/')
                                                          ),
                                                          ceiler=InternalConsistency())


######
# GCR
######
def DicarloMajajHong2015V4GCR_c_lmh():
    return _DicarloMajajHong2015Region_lmh('V4', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=gram_c_regression(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name', train_size=0.5)),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCR_c_lmh():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=gram_c_regression(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name', train_size=0.5)),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCR_c_lmh_ty_01_neg():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=gram_c_regression(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_ty_0.1_neg.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCR_c_lmh_ty_01_pos():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=gram_c_regression(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_ty_0.1_pos.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015ITGCR_c_lmh_tz_01_pos():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=gram_c_regression(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_tz_0.1_pos.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCR_c_lmh_tz_01_neg():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_regression(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_tz_0.1_neg.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015V4GCR_nc_lmh():
    return _DicarloMajajHong2015Region_lmh('V4', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_regression(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name', train_size=0.5)),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCR_nc_lmh():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_regression(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name', train_size=0.5)),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015ITGCR_nc_lmh_ty_01_neg():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_regression(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_ty_0.1_neg.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCR_nc_lmh_ty_01_pos():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_regression(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_ty_0.1_pos.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015ITGCR_nc_lmh_tz_01_pos():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_regression(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_tz_0.1_pos.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCR_nc_lmh_tz_01_neg():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_regression(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_tz_0.1_neg.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015V4GCR_c_lmh():
    return _DicarloMajajHong2015Region_lmh('V4', identifier_metric_suffix='gcr',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_regression(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name', train_size=0.5)),
                                       ceiler=InternalConsistency())

########
# GCPLS
########
def DicarloMajajHong2015V4GCPLS_c_lmh():
    return _DicarloMajajHong2015Region_lmh('V4', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name', train_size=0.5)),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCPLS_c_lmh():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name', train_size=0.5)),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCPLS_c_lmh_ty_01_neg():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_ty_0.1_neg.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCPLS_c_lmh_ty_01_pos():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_ty_0.1_pos.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015ITGCPLS_c_lmh_tz_01_pos():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_tz_0.1_pos.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCPLS_c_lmh_tz_01_neg():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=True), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_tz_0.1_neg.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015V4GCPLS_nc_lmh():
    return _DicarloMajajHong2015Region_lmh('V4', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name', train_size=0.5)),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCPLS_nc_lmh():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name', train_size=0.5)),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCPLS_nc_lmh_ty_01_neg():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_ty_0.1_neg.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCPLS_nc_lmh_ty_01_pos():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_ty_0.1_pos.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCPLS_nc_lmh_tz_01_pos():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_tz_0.1_pos.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())

def DicarloMajajHong2015ITGCPLS_nc_lmh_tz_01_neg():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='gcpls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=old_gram_control_pls(gram_control=False), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                           csv_file ='__majajhonglocal_halves_tz_0.1_neg.csv',
                                           parent_folder = './data/splits/')),
                                       ceiler=InternalConsistency())



########
# Other
########
# def Lore():
#     return _DicarloMajajHong2015Region_lmh_covariate(covariate_image_dir = 'image_dicarlo_hvm_masked', region='IT', identifier_metric_suffix='gcr',
#                                                 similarity_metric=CrossRegressedCorrelationCovariate(
#                                                     regression=covariate_regression(), correlation=pearsonr_correlation(),
#                                                     crossvalidation_kwargs=dict(stratification_coord='object_name', train_size=0.5)),
#                                                 # ceiler=ToleranceConsistency(regression=pls_regression(),
#                                                 #                             correlation=pearsonr_correlation()),
#                                                 ceiler=InternalConsistency(),
#                                                 benchmark_identifier='Lore')


def Lore():
    return _DicarloMajajHong2015Region_lmh_toleranceceiling(region='IT', identifier_metric_suffix='toleranceceiling',
                                                                 similarity_metric=ToleranceCrossValidation(
                                                                     regression=pls_regression(),
                                                                     correlation=pearsonr_correlation(),
                                                                     crossvalidation_kwargs=dict(stratification_coord='object_name')
                                                                 ))


def DicarloMajajHong2015V4Mask_lmh():
    return _DicarloMajajHong2015Region_lmh('V4', identifier_metric_suffix='mask',
                                       similarity_metric=ScaledCrossRegressedCorrelation(
                                           regression=mask_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name',
                                                                       train_size=0.5)),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015ITMask_lmh():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='mask',
                                       similarity_metric=ScaledCrossRegressedCorrelation(
                                           regression=mask_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name', train_size=0.5)),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015V4RDM_lmh():
    return _DicarloMajajHong2015Region_lmh('V4', identifier_metric_suffix='rdm',
                                       similarity_metric=RDMCrossValidated(
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                       train_size=0.5)),
                                       ceiler=RDMConsistency())


def DicarloMajajHong2015ITRDM_lmh():
    return _DicarloMajajHong2015Region_lmh('IT', identifier_metric_suffix='rdm',
                                       similarity_metric=RDMCrossValidated(
                                           crossvalidation_kwargs=dict(stratification_coord='object_name',
                                                                       train_size=0.5)),
                                       ceiler=RDMConsistency())






