from brainscore.utils import LazyLoad
from brainscore.benchmarks.majajhong2015_combined import _DicarloMajajHong2015Region_lmh, \
    _DicarloMajajHong2015Region_lmh_covariate, _DicarloMajajHong2015Region_lmh_toleranceceiling, _DicarloMajajHong2015Region_lmh_imagedir, _DicarloMajajHong2015Region_lmh_covariate_gram, _DicarloMajajHong2015Region_lmh_covariate_cache_features
from brainscore.benchmarks.public_benchmarks import _standard_benchmark
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency, ToleranceConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation
from brainscore.metrics.regression_extra import CrossRegressedCorrelationCovariate, semipartial_regression, \
    semipartial_pls, gram_pls, old_gram_control_regression, old_gram_control_pls, \
    ToleranceCrossValidation, CrossRegressedCorrelationDrew, gram_linear, CrossRegressedCorrelationSemiPartial

from brainscore.benchmarks.majajhong2015 import load_assembly as load_majajhong2015, VISUAL_DEGREES as majajhong2015_degrees, \
    NUMBER_OF_TRIALS as majajhong2015_trials
from brainscore.benchmarks.majajhong2015 import load_assembly as load_majajhong2015, VISUAL_DEGREES as majajhong2015_degrees, \
    BIBTEX as majajhong2015_bibtex
import functools
from brainscore.benchmarks.majajhong2015 import load_assembly as load_majajhong2015
from brainscore.benchmarks.majajhong2015_combined import load_assembly as load_majajhong2015_combined

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

    elif benchmark_identifier == 'tol_extract_features':
        return get_tol_extract_features(crossvalidation_kwargs, **kwargs)

    elif benchmark_identifier == 'tol_semi_partial':
        return get_tol_semi_partial(crossvalidation_kwargs, **kwargs)

    else:
        raise NotImplemented("This tolerance identifier has not been implemented yet")


def _gather_cv_kwargs(**kwargs):
    assert (kwargs.get('baseline', None) is not None)

    if kwargs['baseline']:
        assert (kwargs.get('train_size', None) is not None)
        assert (kwargs.get('test_size', None) is not None)
        assert (kwargs.get('baseline_splits', None) is not None)

        crossvalidation_kwargs = dict(
            stratification_coord='object_name',
            train_size=kwargs['train_size'],
            test_size=kwargs['test_size'],
            splits=int(kwargs['baseline_splits']))

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

    similarity_metric_kwargs = dict(
        covariate_control=kwargs['control'],
        control_regression=gram_linear(gram=kwargs['gram'], pca_kwargs={'n_components':0.99}),
        main_regression=pls_regression(),
        correlation=pearsonr_correlation(),
        crossvalidation_kwargs=crossvalidation_kwargs,
        fname=kwargs.get('explained_variance_fname', None),
        tag=kwargs.get('csv_file', None)
    )

    top_function_kwargs = dict(
        covariate_image_dir=kwargs['covariate_image_dir'],
        region=kwargs.get('region'),
        identifier_metric_suffix='tol_99',
        similarity_metric=CrossRegressedCorrelationDrew(**similarity_metric_kwargs),
        ceiler=InternalConsistency(),
        assembly_name=kwargs.get('assembly_name')
    )



    def top_function():
        return _DicarloMajajHong2015Region_lmh_covariate(**top_function_kwargs)
        # return _DicarloMajajHong2015Region_lmh_covariate(
        #     covariate_image_dir=kwargs['covariate_image_dir'],
        #     region='IT', identifier_metric_suffix='Drew',
        #     similarity_metric=CrossRegressedCorrelationDrew(
        #         covariate_control=kwargs['control'],
        #         control_regression=gram_linear(gram=kwargs['gram'], pca_kwargs={'n_components': 0.99}),
        #         main_regression=pls_regression(),
        #         correlation=pearsonr_correlation(),
        #         crossvalidation_kwargs=crossvalidation_kwargs,
        #         fname=kwargs.get('explained_variance_fname', None),
        #         tag=kwargs.get('csv_file', None)),
        #     ceiler=InternalConsistency()
        # )

    return LazyLoad(top_function)


def get_tol_extract_features(crossvalidation_kwargs, **kwargs):
    '''
    Similar to tol_drew, except the control regression is PCA + linear regression instead of PLS, and with all components
    so we make sure all of the control variable is regressed out
    '''

    assert (kwargs.get('covariate_image_dir', None) is not None)
    assert (kwargs.get('control', None) is not None)
    assert (kwargs.get('gram', None) is not None)

    similarity_metric_kwargs = dict(
        covariate_control=kwargs['control'],
        control_regression=gram_linear(gram=kwargs['gram'], pca_kwargs={'n_components':0.99}),
        main_regression=pls_regression(),
        correlation=pearsonr_correlation(),
        crossvalidation_kwargs=crossvalidation_kwargs,
        fname=kwargs.get('explained_variance_fname', None),
        tag=kwargs.get('csv_file', None)
    )

    top_function_kwargs = dict(
        covariate_image_dir=kwargs['covariate_image_dir'],
        region=kwargs.get('region'),
        identifier_metric_suffix='tol_99',
        similarity_metric=CrossRegressedCorrelationDrew(**similarity_metric_kwargs),
        ceiler=InternalConsistency(),
        assembly_name=kwargs.get('assembly_name')
    )



    def top_function():
        return _DicarloMajajHong2015Region_lmh_covariate_cache_features(**top_function_kwargs)
        # return _DicarloMajajHong2015Region_lmh_covariate(
        #     covariate_image_dir=kwargs['covariate_image_dir'],
        #     region='IT', identifier_metric_suffix='Drew',
        #     similarity_metric=CrossRegressedCorrelationDrew(
        #         covariate_control=kwargs['control'],
        #         control_regression=gram_linear(gram=kwargs['gram'], pca_kwargs={'n_components': 0.99}),
        #         main_regression=pls_regression(),
        #         correlation=pearsonr_correlation(),
        #         crossvalidation_kwargs=crossvalidation_kwargs,
        #         fname=kwargs.get('explained_variance_fname', None),
        #         tag=kwargs.get('csv_file', None)),
        #     ceiler=InternalConsistency()
        # )

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

# def get_tol_99_(crossvalidation_kwargs, **kwargs):
#     '''
#     This should use the the same NeuralBenchMark class as the one you'd get with the dicarlo.MajajHong2015public.IT-pls
#     identifier, except with more flexibility in arguments you can pass
#     '''
#
#     def top_function():
#         return _standard_benchmark('dicarlo.MajajHong2015.IT.public',
#                             load_assembly=functools.partial(load_majajhong2015, region='IT', access='public'),
#                             visual_degrees=majajhong2015_degrees, number_of_trials=majajhong2015_trials,
#                             stratification_coord='object_name', bibtex=majajhong2015_bibtex)
#
#     return LazyLoad(top_function)



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
    proposed, which should be similar in spirit to a semi-partial correlation. It's also using PLS instead
    of linear regression + PCA
    '''

    assert (kwargs.get('covariate_image_dir', None) is not None)
    assert (kwargs.get('control', None) is not None)
    assert (kwargs.get('gram', None) is not None)

    similarity_metric_kwargs = dict(
        covariate_control=kwargs['control'],
        control_regression=gram_linear(gram=kwargs['gram'], with_pca=True, pca_kwargs={'n_components':25}),
        main_regression=pls_regression(),
        correlation=pearsonr_correlation(),
        crossvalidation_kwargs=crossvalidation_kwargs,
        fname=kwargs.get('explained_variance_fname', None),
        tag=kwargs.get('csv_file', None)
    )

    top_function_kwargs = dict(
        covariate_image_dir=kwargs['covariate_image_dir'],
        region=kwargs.get('region'),
        identifier_metric_suffix='tol_99',
        similarity_metric=CrossRegressedCorrelationSemiPartial(**similarity_metric_kwargs),
        ceiler=InternalConsistency(),
        assembly_name=kwargs.get('assembly_name')
    )

    def top_function():
        return _DicarloMajajHong2015Region_lmh_covariate(**top_function_kwargs)
        # return _DicarloMajajHong2015Region_lmh_covariate(
        #     covariate_image_dir=kwargs['covariate_image_dir'],
        #     region='IT', identifier_metric_suffix='Drew',
        #     similarity_metric=CrossRegressedCorrelationDrew(
        #         covariate_control=kwargs['control'],
        #         control_regression=gram_linear(gram=kwargs['gram'], pca_kwargs={'n_components': 0.99}),
        #         main_regression=pls_regression(),
        #         correlation=pearsonr_correlation(),
        #         crossvalidation_kwargs=crossvalidation_kwargs,
        #         fname=kwargs.get('explained_variance_fname', None),
        #         tag=kwargs.get('csv_file', None)),
        #     ceiler=InternalConsistency()
        # )

    return LazyLoad(top_function)


    return None



