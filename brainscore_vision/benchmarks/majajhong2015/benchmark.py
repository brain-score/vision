# need to access classes and functionality in public namespace majajhong


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
