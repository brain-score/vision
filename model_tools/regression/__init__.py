from model_tools.xarray_utils import XarrayRegression, Defaults as XarrayDefaults
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression


def pls_regression(n_components=25, expected_dims=XarrayDefaults.expected_dims,
                   neuroid_dim=XarrayDefaults.neuroid_dim, neuroid_coord=XarrayDefaults.neuroid_coord):
    regression = PLSRegression(n_components=n_components, scale=False)
    regression = XarrayRegression(regression, expected_dims=expected_dims,
                                  neuroid_dim=neuroid_dim, neuroid_coord=neuroid_coord)
    return regression


def linear_regression(expected_dims=XarrayDefaults.expected_dims,
                      neuroid_dim=XarrayDefaults.neuroid_dim, neuroid_coord=XarrayDefaults.neuroid_coord):
    regression = LinearRegression()
    regression = XarrayRegression(regression, expected_dims=expected_dims,
                                  neuroid_dim=neuroid_dim, neuroid_coord=neuroid_coord)
    return regression
