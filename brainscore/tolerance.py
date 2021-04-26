from brainscore.utils import LazyLoad
from brainscore.benchmarks.majajhong2015_combined import _DicarloMajajHong2015Region_lmh, \
    _DicarloMajajHong2015Region_lmh_covariate, _DicarloMajajHong2015Region_lmh_masked, \
    _DicarloMajajHong2015Region_lmh_toleranceceiling
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency, ToleranceConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, gram_control_regression, gram_control_pls, pearsonr_correlation
from brainscore.metrics.regression_extra import CrossRegressedCorrelationCovariate, semipartial_regression, \
    semipartial_pls, \
    ToleranceCrossValidation, CrossRegressedCorrelationDrew


def get_benchmark(benchmark_identifier, **kwargs):
    # Check arguments (need to exist and not be None)
    assert (kwargs.get('baseline', None) is not None)

    # Sort out the crossvalidation kwargs
    if kwargs['baseline']:
        assert (kwargs.get('train_size', None) is not None)
        assert (kwargs.get('test_size', None) is not None)

        crossvalidation_kwargs = dict(
            stratification_coord='object_name',
            train_size=kwargs['train_size'],
            test_size=kwargs['test_size'])
    else:
        assert (kwargs.get('parent_folder', None) is not None)
        assert (kwargs.get('csv_file', None) is not None)

        crossvalidation_kwargs = dict(
            stratification_coord='object_name',
            parent_folder=kwargs['parent_folder'],
            csv_file=kwargs['csv_file'])

    # Get the right benchmark function
    if benchmark_identifier == 'tol_drew':
        assert (kwargs.get('covariate_image_dir', None) is not None)
        assert (kwargs.get('control', None) is not None)

        def top_function():
            return _DicarloMajajHong2015Region_lmh_covariate(
                covariate_image_dir=kwargs['covariate_image_dir'],
                region='IT', identifier_metric_suffix='Drew',
                similarity_metric=CrossRegressedCorrelationDrew(
                    covariate_control=kwargs['control'],
                    regression=pls_regression(),
                    correlation=pearsonr_correlation(),
                    crossvalidation_kwargs=crossvalidation_kwargs),
                ceiler=InternalConsistency()
            )
    else:
        raise NotImplemented("This tolerance identifier has not been implemented yet")

    return LazyLoad(top_function)
