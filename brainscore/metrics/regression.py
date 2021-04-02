import scipy.stats
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from brainio_base.assemblies import walk_coords
from brainscore.metrics.mask_regression import MaskRegression
from brainscore.metrics.transformations import CrossValidation
from .xarray_utils import XarrayRegression, XarrayCorrelation

from collections import Counter
from collections import OrderedDict



class CrossRegressedCorrelation:
    def __init__(self, regression, correlation, crossvalidation_kwargs=None):
        regression = regression or pls_regression()
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidation(**crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation

    def __call__(self, source, target):
        return self.cross_validation(source, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train, target_train, source_test, target_test):
        self.regression.fit(source_train, target_train)
        prediction = self.regression.predict(source_test)
        score = self.correlation(prediction, target_test)
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')


class ScaledCrossRegressedCorrelation:
    def __init__(self, *args, **kwargs):
        self.cross_regressed_correlation = CrossRegressedCorrelation(*args, **kwargs)
        self.aggregate = self.cross_regressed_correlation.aggregate

    def __call__(self, source, target):
        scaled_values = scale(target, copy=True)
        target = target.__class__(scaled_values, coords={
            coord: (dims, value) for coord, dims, value in walk_coords(target)}, dims=target.dims)
        return self.cross_regressed_correlation(source, target)


class SingleRegression():
    def __init__(self):
        self.mapping = []

    def fit(self, X, Y):
        X = X.values
        Y = Y.values
        n_stim, n_neuroid = X.shape
        _, n_neuron = Y.shape
        r = np.zeros((n_neuron, n_neuroid))
        for neuron in range(n_neuron):
            r[neuron, :] = pearsonr(X, Y[:, neuron:neuron+1])
        self.mapping = np.nanargmax(r, axis=1)

    def predict(self, X):
        X = X.values
        Ypred = X[:, self.mapping]
        return Ypred


class GramControlRegression():
    def __init__(self, gram_control=False, channel_coord=None, scaler_kwargs=None, pca_kwargs=None, regression_kwargs=None):
        self.gram_control = gram_control
        self.channel_coord = channel_coord
        self.scaler_kwargs = scaler_kwargs or {}
        self.pca_kwargs = pca_kwargs or {}
        self.regression_kwargs = regression_kwargs or {}
        self.scaler_x = StandardScaler(**self.scaler_kwargs)
        self.scaler_y = StandardScaler(**self.scaler_kwargs)
        self.pca_x = PCA(**self.pca_kwargs)
        self.pca_gram = PCA(**self.pca_kwargs)
        self.control_regression = LinearRegression(**self.regression_kwargs)
        self.main_regression = LinearRegression(**self.regression_kwargs)

    def _unflatten(self, X, channel_coord=None):
        """
        Unflattens NeuroidAssembly of flattened model activations to
        BxCxH*W (or BXCxW*H not sure, also not sure if it matters)

        Using the information in coordinates channel, channel_x, channel_y which give the index along each
        of the original axes

        Order of coordinates (which one represents first axis, second, third) is determined by checking which one's
        values change slowest (i.e., reverse sort by first occurence of a 1-value)
        """

        # Get coord:(original axis length, first occurence of 1
        X_shape = OrderedDict({
            'channel': (X.channel.values.max()+1, np.where(X.channel.values == 1)[0][0]),
            'channel_x': (X.channel_x.values.max()+1, np.where(X.channel_x.values == 1)[0][0]),
            'channel_y': (X.channel_y.values.max()+1, np.where(X.channel_y.values == 1)[0][0])
        })

        # Determine which coordinate represents channels
        if self.channel_coord == None:
            # Hacky attempt to determine automatically (usually W==H != C)
            X_axis_sizes = [i[0] for i in X_shape.values()]
            frequencies = {key:Counter(X_axis_sizes)[value[0]] for key, value in X_shape.items()}
            if sorted(list(frequencies.values())) != [1,2,2]:
                raise ValueError('channel_coord is None and failed to automatically determine it')
            else:
                channel_coord = [key for key, value in frequencies.items() if value==1][0]

        # Sort coordinates such that first one represents first axis in original matrix, etc.
        X_shape = OrderedDict(sorted(X_shape.items(), key=lambda x: x[1][1], reverse=True))

        # Unflatten X
        B = X.shape[0]
        reshape_to = [B] + [value[0] for key, value in X_shape.items()]
        X = X.values.reshape(reshape_to)

        # Channels first and reshape to BxCxH*W (or W*H, not sure)
        channel_index = [i for i, (key, value) in enumerate(X_shape.items()) if key==channel_coord][0]
        channel_index = channel_index + 1 # bc very first is B
        transpose_to = [0, channel_index]+ [i for i in [1,2,3] if i != channel_index]
        X = np.transpose(X, transpose_to)
        X = X.reshape(list(X.shape[0:2])+[-1])

        return X

    def _preprocess_gram(self, X, fit=True):
        # Center/scale
        if fit:
            self.scaler_x.fit(X)
        X.values = self.scaler_x.transform(X)

        # Compute gram matrices
        X_grams = self._unflatten(X, self.channel_coord) # Unflatten X to BxCxH*W (or W*H, not sure)
        X_grams = np.einsum("ijk, ikl -> ijl", X_grams, np.transpose(X_grams, [0,2,1]))
        #X_grams = X_grams/X.size # is this the right normalization?
        X_grams = X_grams.reshape(X_grams.shape[0], -1)

        # PCA
        if fit:
            self.pca_gram.fit(X_grams)
            self.pca_x.fit(X)
        X_grams = self.pca_gram.transform(X_grams)
        X = self.pca_x.transform(X)

        # Residuals
        if fit:
            self.control_regression.fit(X_grams, X)
        X = X - self.control_regression.predict(X_grams)

        return X

    def _preprocess(self, X, fit=True):
        if fit:
            self.scaler_x.fit(X)
            self.pca_x.fit(X)
        X = self.scaler_x.transform(X)
        X = self.pca_x.transform(X)

        return X


    def fit(self, X, Y):
        if self.gram_control:
            X = self._preprocess_gram(X, fit=True)
        else:
            X = self._preprocess(X, fit=True)

        Y = self.scaler_y.fit_transform(Y)
        self.main_regression.fit(X, Y)


    def predict(self, X):
        if self.gram_control:
            X = self._preprocess_gram(X, fit=False)
        else:
            X = self._preprocess(X, fit=False)

        Ypred = self.main_regression.predict(X)
        return self.scaler_y.inverse_transform(Ypred) # is this wise?


def mask_regression():
    regression = MaskRegression()
    regression = XarrayRegression(regression)
    return regression


def pls_regression(regression_kwargs=None, xarray_kwargs=None):
    regression_defaults = dict(n_components=25, scale=False)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression = PLSRegression(**regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression

def gram_control_regression(gram_control, channel_coord=None, scaler_kwargs=None, pca_kwargs=None, regression_kwargs=None, xarray_kwargs=None):
    scaler_defaults = dict(with_std=False)
    pca_defaults = dict(n_components=25)
    scaler_kwargs = {**scaler_defaults, **(scaler_kwargs or {})}
    pca_kwargs = {**pca_defaults, **(pca_kwargs or {})}
    regression_kwargs = regression_kwargs or {}
    regression = GramControlRegression(gram_control=gram_control,
                                       channel_coord=None,
                                       scaler_kwargs=scaler_kwargs,
                                       pca_kwargs=pca_kwargs,
                                       regression_kwargs=regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression



def linear_regression(xarray_kwargs=None):
    regression = LinearRegression()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression

def ridge_regression(xarray_kwargs=None):
    regression = Ridge()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def pearsonr_correlation(xarray_kwargs=None):
    xarray_kwargs = xarray_kwargs or {}
    return XarrayCorrelation(scipy.stats.pearsonr, **xarray_kwargs)


def single_regression(xarray_kwargs=None):
    regression = SingleRegression()
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def pearsonr(x, y):
    xmean = x.mean(axis=0, keepdims=True)
    ymean = y.mean(axis=0, keepdims=True)

    xm = x - xmean
    ym = y - ymean

    normxm = scipy.linalg.norm(xm, axis=0, keepdims=True)
    normym = scipy.linalg.norm(ym, axis=0, keepdims=True)

    r = ((xm/normxm)*(ym/normym)).sum(axis=0)

    return r
