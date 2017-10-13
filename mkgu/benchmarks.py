import mkgu
from mkgu import metrics


def neural_fit_hvm_it_jonas_20171011(model):
    neural = mkgu.get_assembly(name='HvM')
    hvmit = neural.sel(region='IT')
    df = metrics.neural_fit(hvmit, model)
    return df.groupby('site').mean().median()


def neural_fit_hvm_it_time_jonas_20171011(model):
    neural = mkgu.get_assembly(name='HvM')
    hvmit = neural.sel(region='IT')
    for tp in hvmit:
        df = metrics.neural_fit(hvmit.sel(time_bins=tp), model)
    return df.groupby('site').mean().median()
