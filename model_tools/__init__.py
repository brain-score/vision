from model_tools.regression import linear_regression, pls_regression

mapping_pool = {
    'linear_regression': linear_regression(),
    'pls_regression-25': pls_regression(n_components=25)
}
