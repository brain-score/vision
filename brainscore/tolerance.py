from brainscore.utils import LazyLoad
from brainscore.benchmarks.majajhong2015_combined import _DicarloMajajHong2015Region_lmh, \
    _DicarloMajajHong2015Region_lmh_covariate, _DicarloMajajHong2015Region_lmh_masked, \
    _DicarloMajajHong2015Region_lmh_toleranceceiling, _DicarloMajajHong2015Region_lmh_imagedir, _DicarloMajajHong2015Region_lmh_covariate_gram
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency, ToleranceConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation
from brainscore.metrics.regression_extra import CrossRegressedCorrelationCovariate, semipartial_regression, \
    semipartial_pls, gram_pls, old_gram_control_regression, old_gram_control_pls, \
    ToleranceCrossValidation, CrossRegressedCorrelationDrew, gram_linear


def get_benchmark(benchmark_identifier, **kwargs):
    crossvalidation_kwargs = _gather_cv_kwargs(**kwargs)

    # Get the right benchmark function
    if benchmark_identifier == 'tol_drew':
        return get_tol_drew(crossvalidation_kwargs, **kwargs)

    elif benchmark_identifier == 'tol_99':
        return get_tol_99(crossvalidation_kwargs, **kwargs)

    elif benchmark_identifier == 'tol_objects':
        return get_tol_imagedir(crossvalidation_kwargs, **kwargs)

    elif benchmark_identifier == 'tol_99_gram':
        return get_tol_99_gram(crossvalidation_kwargs, **kwargs)

    else:
        raise NotImplemented("This tolerance identifier has not been implemented yet")


def _gather_cv_kwargs(**kwargs):
    assert (kwargs.get('baseline', None) is not None)

    if kwargs['baseline']:
        assert (kwargs.get('train_size', None) is not None)
        assert (kwargs.get('test_size', None) is not None)

        crossvalidation_kwargs = dict(
            stratification_coord='object_name',
            train_size=kwargs['train_size'],
            test_size=kwargs['test_size'],
            splits=1)

    else:
        assert (kwargs.get('parent_folder', None) is not None)
        assert (kwargs.get('csv_file', None) is not None)

        crossvalidation_kwargs = dict(
            stratification_coord='object_name',
            parent_folder=kwargs['parent_folder'],
            csv_file=kwargs['csv_file'])

    return crossvalidation_kwargs


def get_tol_drew(crossvalidation_kwargs, **kwargs):
    '''
    Choose this if you want to regress out model activations for an edited set of images in the way Drew and Thomas
    proposed
    '''

    assert (kwargs.get('covariate_image_dir', None) is not None)
    assert (kwargs.get('control', None) is not None)
    assert (kwargs.get('gram', None) is not None)

    def top_function():
        return _DicarloMajajHong2015Region_lmh_covariate(
            covariate_image_dir=kwargs['covariate_image_dir'],
            region='IT', identifier_metric_suffix='Drew',
            similarity_metric=CrossRegressedCorrelationDrew(
                covariate_control=kwargs['control'],
                control_regression=gram_pls() if kwargs['gram'] else pls_regression(),
                main_regression=pls_regression(),
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=crossvalidation_kwargs,
                fname=kwargs.get('explained_variance_fname', None),
                tag=kwargs.get('csv_file', None)),
            ceiler=InternalConsistency()
        )

    return LazyLoad(top_function)


def get_tol_99(crossvalidation_kwargs, **kwargs):
    '''
    Similar to tol_drew, except the control regression is PCA + linear regression instead of PLS, and with all components
    so we make sure all of the control variable is regressed out
    '''

    assert (kwargs.get('covariate_image_dir', None) is not None)
    assert (kwargs.get('control', None) is not None)
    assert (kwargs.get('gram', None) is not None)

    def top_function():
        return _DicarloMajajHong2015Region_lmh_covariate(
            covariate_image_dir=kwargs['covariate_image_dir'],
            region='IT', identifier_metric_suffix='Drew',
            similarity_metric=CrossRegressedCorrelationDrew(
                covariate_control=kwargs['control'],
                control_regression=gram_linear(gram=kwargs['gram']),
                main_regression=pls_regression(),
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=crossvalidation_kwargs,
                fname=kwargs.get('explained_variance_fname', None),
                tag=kwargs.get('csv_file', None)),
            ceiler=InternalConsistency()
        )

    return LazyLoad(top_function)


def get_tol_99_gram(crossvalidation_kwargs, **kwargs):
    '''
    Similar to tol_drew, except the control regression is PCA + linear regression instead of PLS, and with all components
    so we make sure all of the control variable is regressed out
    '''

    assert (kwargs.get('covariate_image_dir', None) is not None)
    assert (kwargs.get('control', None) is not None)
    assert (kwargs.get('gram', None) is not None)

    def top_function():
        return _DicarloMajajHong2015Region_lmh_covariate_gram(
            covariate_image_dir=kwargs['covariate_image_dir'],
            gram=kwargs.get('gram', False),
            region='IT', identifier_metric_suffix='Drew',
            similarity_metric=CrossRegressedCorrelationDrew(
                covariate_control=kwargs['control'],
                control_regression=gram_linear(gram=False),  # Gram will already be computed at higher level of the code
                main_regression=pls_regression(),
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=crossvalidation_kwargs,
                fname=kwargs.get('explained_variance_fname', None),
                tag=kwargs.get('csv_file', None)),
            ceiler=InternalConsistency()
        )

    return LazyLoad(top_function)


def get_tol_imagedir(crossvalidation_kwargs, **kwargs):
    '''
    Choose this when you want to use activations for edited MajajHong2015 stimuli to predict the MajajHong2015 (non-edited)
    neural data. For example, we had one analysis where we passed images stripped off the background (only object remained)
    to the CNN.
    Make sure the edited images in the image_dir have the same filenames as their respective counterpart in the original
    stimulus set.
    '''

    assert (kwargs.get('image_dir', None) is not None)
    assert (kwargs.get('control', None) is not None)

    def top_function():
        return _DicarloMajajHong2015Region_lmh_imagedir(
            image_dir=kwargs['image_dir'],
            region='IT',
            identifier_metric_suffix='pls',
            similarity_metric=CrossRegressedCorrelation(
                regression=pls_regression(),
                correlation=pearsonr_correlation(),
                crossvalidation_kwargs=crossvalidation_kwargs),
            ceiler=InternalConsistency())

    return LazyLoad(top_function)


def get_tol_semi_partial(crossvalidation_kwargs, **kwargs):
    '''
    Choose this if you want to regress out model activations for an edited set of images in the way Lore
    proposed, which should be similar in spirit to a semi-partial correlation.
    '''

    return None



