import scipy.stats
import numpy as np
import os
import time
import torch

from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from brainio_base.assemblies import walk_coords
from brainscore.metrics.mask_regression import MaskRegression
from brainscore.metrics.transformations import CrossValidation
from brainscore.metrics.transformations_extra import ToleranceCrossValidation
from brainscore.metrics.transformations_extra import CrossValidationCustomPlusBaseline
from brainscore.metrics.regression import pls_regression
from .xarray_utils import XarrayRegression, XarrayCorrelation

from .xarray_utils import Defaults
from brainio_base.assemblies import NeuroidAssembly, array_is_element, walk_coords
from copy import deepcopy
import xarray as xr
from collections import OrderedDict, Counter
from sklearn.metrics import explained_variance_score
import pandas as pd
from einsumt import einsumt as einsum

class InternalCrossedRegressedCorrelation:
    def __init__(self, regression, correlation, crossvalidation_kwargs=None):
        regression = regression or pls_regression()
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = ToleranceCrossValidation(**crossvalidation_kwargs)
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


class CrossRegressedCorrelationCovariate:
    def __init__(self, regression, correlation, crossvalidation_kwargs=None):
        # regression = regression or pls_regression()
        crossvalidation_defaults = dict(train_size=.9, test_size=None)
        crossvalidation_kwargs = {**crossvalidation_defaults, **(crossvalidation_kwargs or {})}

        self.cross_validation = CrossValidation(expecting_coveriate=True, **crossvalidation_kwargs)
        self.regression = regression
        self.correlation = correlation

    def __call__(self, source, covariate, target):
        return self.cross_validation(source, covariate, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train, covariate_train, target_train, source_test, covariate_test, target_test):
        self.regression.fit(source_train, covariate_train, target_train)
        prediction = self.regression.predict(source_test, covariate_test)
        score = self.correlation(prediction, target_test)
        return score

    def aggregate(self, scores):
        return scores.median(dim='neuroid')


class CrossRegressedCorrelationDrew:
    def __init__(self, main_regression, control_regression, correlation, covariate_control=True, fname=None, tag=None, crossvalidation_kwargs=None):
        # regression = regression or pls_regression()
        self.crossvalidation_kwargs = crossvalidation_kwargs or {}

        self.cross_validation = CrossValidation(expecting_coveriate=True, **self.crossvalidation_kwargs)
        self.main_regression = main_regression
        self.control_regression = control_regression
        self.correlation = correlation
        self.covariate_control = covariate_control
        self.fname = fname
        self.tag = tag  # just an extra tag to help keep track of what we're running and write it to fname. Doesn't have
                        # any impact beyond the what is written to fname


    def __call__(self, source, covariate, target):
        return self.cross_validation(source, covariate, target, apply=self.apply, aggregate=self.aggregate)

    def apply(self, source_train, covariate_train, target_train, source_test, covariate_test, target_test):
        # vv Hacky but the model activation assemblies don't have the stimulus index and it makes the alignment fail
        # vv All we need for alignment is image_id and neuroid anyway (and perhaps repetition in some cases)
        target_train = target_train.reset_index('stimulus', drop=True)
        target_test = target_test.reset_index('stimulus', drop=True)

        X1_train = source_train
        X2_train = covariate_train
        Y_train = target_train

        X1_test = source_test
        X2_test = covariate_test
        Y_test = target_test

        if self.covariate_control:
            # FIT (train)
            ##############

            # 1) Regressing on the covariate (X2)
            self.control_regression.fit(X2_train, Y_train)

            # Residualize Y
            Y_pred_train = self.control_regression.predict(X2_train)
            Y_train, Y_pred_train = xr.align(Y_train, Y_pred_train)
            assert (np.array_equal(Y_train.image_id.values, Y_pred_train.image_id.values))

            Y_residuals_train = Y_train - Y_pred_train

            # 2) Regressing the residuals on the source (X1)
            self.main_regression.fit(X1_train, Y_residuals_train)

            if self.fname:
                dict_to_save = {}
                dict_to_save['train'] = True
                dict_to_save['explained_variance_control'] = explained_variance_score(Y_train, Y_pred_train)
                dict_to_save['similarity_control'] = self.correlation(Y_pred_train, Y_train).median().item()
                dict_to_save['model'] = X1_train.model.values[0]
                dict_to_save['layer'] = X1_train.layer.values[0]
                dict_to_save['covariate_identifier'] = X2_train.stimulus_set_identifier
                self.write_to_file(dict_to_save, self.fname)



            # PREDICTION (test)
            ####################

            # Residualize Y (wo refitting)
            Y_pred_test = self.control_regression.predict(X2_test)
            Y_test, Y_pred_test = xr.align(Y_test, Y_pred_test)
            assert (np.array_equal(Y_test.image_id.values, Y_pred_test.image_id.values))

            Y_residuals_test = Y_test - Y_pred_test

            # Get predicted residuals and correlate to test residuals
            prediction = self.main_regression.predict(X1_test)

            # vv feels a little weird to me that neither are ground truth (both are result of soem regression)
            # vv we're no longer comparing directly to neural data, but to residuals of neural data, which feels like a big deviation from the original pipeline
            score = self.correlation(prediction, Y_residuals_test)

            if self.fname:
                dict_to_save = {}
                dict_to_save['train'] = False
                dict_to_save['explained_variance_control'] = explained_variance_score(Y_test, Y_pred_test)
                dict_to_save['similarity_control'] = self.correlation(Y_pred_test, Y_test).median().item()
                dict_to_save['model'] = X1_test.model.values[0]
                dict_to_save['layer'] = X1_test.layer.values[0]
                dict_to_save['covariate_identifier'] = X2_test.stimulus_set_identifier
                self.write_to_file(dict_to_save, self.fname)

        else:
            # FIT (train)
            self.main_regression.fit(X1_train, Y_train)
            # PREDICT (test)
            Y_pred = self.main_regression.predict(X1_test)
            score = self.correlation(Y_pred, Y_test)

        return score

    def write_to_file(self, dict_to_save, fname):
        dict_to_save['tag'] = self.tag
        dict_to_save['class'] = self.__class__.__name__
        dict_to_save['control_regression'] = self.control_regression._regression.__class__.__name__
        dict_to_save['main_regression'] = self.main_regression._regression.__class__.__name__
        dict_to_save['csv_file'] = self.crossvalidation_kwargs.get('csv_file', None)
        dict_to_save['baseline'] = False if dict_to_save['csv_file'] else True
        dict_to_save['gram'] = self.control_regression._regression.gram if hasattr(self.control_regression._regression, 'gram') else None
        dict_to_save['control'] = self.covariate_control

        df_to_save = pd.DataFrame(dict_to_save, index=[0])
        with open(fname, 'a') as f:
            df_to_save.to_csv(f, mode='a', header=f.tell() == 0)



    def aggregate(self, scores):
        return scores.median(dim='neuroid')


# class DrewPLS():
#     def __init__(self, covariate_control = False, regression_kwargs=None):
#         self.covariate_control = covariate_control
#         self.regression_kwargs = regression_kwargs or {}
#         self.control_regression = PLSRegression(**self.regression_kwargs)
#         self.main_regression = PLSRegression(**self.regression_kwargs)
#
#     def _get_residuals(self, X, X_cov, fit=True):
#         # Residuals
#         if fit:
#             self.control_regression.fit(X_cov, X)
#         X = X - self.control_regression.predict(X_cov)
#
#         return X
#
#     def fit(self, X, X_cov, Y):
#
#         # Residuals
#         if self.covariate_control:
#             X = self._get_residuals(X, X_cov, fit=True)
#
#         self.main_regression.fit(X, Y)
#
#     def predict(self, X, X_cov):
#
#         # Residuals
#         if self.covariate_control:
#             X = self._get_residuals(X, X_cov, fit=False)
#
#         Ypred = self.main_regression.predict(X)
#         return Ypred


class SemiPartialRegression():
    def __init__(self, covariate_control=False, scaler_kwargs=None, pca_kwargs=None, regression_kwargs=None):
        self.covariate_control = covariate_control
        self.scaler_kwargs = scaler_kwargs or {}
        self.pca_kwargs = pca_kwargs or {}
        self.regression_kwargs = regression_kwargs or {}
        self.scaler_x = StandardScaler(**self.scaler_kwargs)
        self.scaler_cov = StandardScaler(**self.scaler_kwargs)
        self.scaler_y = StandardScaler(**self.scaler_kwargs)
        self.pca_x = PCA(**self.pca_kwargs)
        self.pca_cov = PCA(**self.pca_kwargs)
        self.control_regression = LinearRegression(**self.regression_kwargs)
        self.main_regression = LinearRegression(**self.regression_kwargs)

    def _get_residuals(self, X, X_cov, fit=True):
        # Residuals
        if fit:
            self.control_regression.fit(X_cov, X)
        X = X - self.control_regression.predict(X_cov)

        return X

    def fit(self, X, X_cov, Y):
        # Center/scale
        X.values = self.scaler_x.fit_transform(X)
        X_cov.values = self.scaler_cov.fit_transform(X_cov)
        Y = self.scaler_y.fit_transform(Y)

        # PCA
        X = self.pca_x.fit_transform(X)
        X_cov = self.pca_cov.fit_transform(X_cov)

        # Residuals
        if self.covariate_control:
            X = self._get_residuals(X, X_cov, fit=True)

        self.main_regression.fit(X, Y)

    def predict(self, X, X_cov):
        # Center/scale
        X.values = self.scaler_x.transform(X)
        X_cov.values = self.scaler_cov.transform(X_cov)

        # PCA
        X = self.pca_x.transform(X)
        X_cov = self.pca_cov.transform(X_cov)

        # Residuals
        if self.covariate_control:
            X = self._get_residuals(X, X_cov, fit=False)

        Ypred = self.main_regression.predict(X)
        return self.scaler_y.inverse_transform(Ypred)  # is this wise?


class SemiPartialPLS():
    def __init__(self, covariate_control=False, regression_kwargs=None):
        self.covariate_control = covariate_control
        self.regression_kwargs = regression_kwargs or {}
        self.control_regression = PLSRegression(**self.regression_kwargs)
        self.main_regression = PLSRegression(**self.regression_kwargs)

    def _get_residuals(self, X, X_cov, fit=True):
        # Residuals
        if fit:
            self.control_regression.fit(X_cov, X)
        X = X - self.control_regression.predict(X_cov)

        return X

    def fit(self, X, X_cov, Y):

        # Residuals
        if self.covariate_control:
            X = self._get_residuals(X, X_cov, fit=True)

        self.main_regression.fit(X, Y)

    def predict(self, X, X_cov):

        # Residuals
        if self.covariate_control:
            X = self._get_residuals(X, X_cov, fit=False)

        Ypred = self.main_regression.predict(X)
        return Ypred


class GramPLS():
    def __init__(self, channel_coord=None, regression_kwargs=None):
        self.channel_coord = channel_coord
        self.regression_kwargs = regression_kwargs or {}
        self.regression = PLSRegression(**self.regression_kwargs)

    def fit(self, X, Y):
        t = time.time()
        X = unflatten(X, channel_coord=self.channel_coord)
        X = X.reshape(list(X.shape[0:2]) + [-1])
        X = take_gram(X)
        print('getting gram took ', str(time.time()-t))
        self.regression.fit(X, Y)

    def predict(self, X):
        t = time.time()
        X = unflatten(X, channel_coord=self.channel_coord)
        # Reshape to BxCxH*W (or W*H, not sure)
        X = X.reshape(list(X.shape[0:2]) + [-1])
        X = take_gram(X)
        print('getting gram took ', str(time.time() - t))
        Y_pred = self.regression.predict(X)
        return Y_pred


class GramLinearRegression():
    def __init__(self, gram=True, channel_coord=None, scaler_kwargs=None, pca_kwargs=None, regression_kwargs=None):
        self.channel_coord = channel_coord
        self.regression_kwargs = regression_kwargs or {}
        self.pca_kwargs = pca_kwargs or {}
        self.scaler_kwargs = scaler_kwargs or {}
        self.regression = LinearRegression(**self.regression_kwargs)
        self.pca = PCA(**self.pca_kwargs)
        self.scaler = StandardScaler(**self.scaler_kwargs)
        self.gram = gram

    def fit(self, X, Y):
        print('FITTING')
        if self.gram:
            t = time.time()
            X = unflatten(X, channel_coord=self.channel_coord)
            X = X.reshape(list(X.shape[0:2]) + [-1])
            X = take_gram(X)
            print('Getting gram took ', str(time.time() - t))
            print('Gram shape is ', X.shape)

        # Scale
        t = time.time()
        X = self.scaler.fit_transform(X)
        print('Fitting scaler took ', str(time.time() - t))

        # PCA
        t = time.time()
        #X_wo_gpu = self.pca.fit_transform(X)
        #del(X_wo_gpu)
        print('PCA took ', str(time.time() - t))

        import pdb; pdb.set_trace()

        # PCA GPU
        t = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.from_numpy(X)
        X = X.to(device=device)
        u, s, v = torch.svd(X)
        del(u)
        del(s)
        X = torch.matmul(X, v)
        X = X.cpu().numpy()
        n_components = np.argmax(np.cumsum(X.var(axis=0))/np.sum(X.var(axis=0))>=0.99) + 1
        X = X[:,0:n_components]
        print('PCA with gpu took ', str(time.time() - t))



        # Regression
        t = time.time()
        self.regression.fit(X, Y)
        print('Fitting regression took ', str(time.time() - t))

    def predict(self, X):
        print('PREDICTING')
        if self.gram:
            t = time.time()
            X = unflatten(X, channel_coord=self.channel_coord)
            # Reshape to BxCxH*W (or W*H, not sure)
            X = X.reshape(list(X.shape[0:2]) + [-1])
            X = take_gram(X)
            print('Getting gram took ', str(time.time() - t))
            print('Gram shape is ', X.shape)

        # Scaling
        t = time.time()
        X = self.scaler.transform(X)
        print('Running scaler took ', str(time.time() - t))


        # PCA
        t = time.time()
        X = self.pca.transform(X)
        print('PCA took ', str(time.time() - t))

        # Regression
        t = time.time()
        Y_pred = self.regression.predict(X)
        print('Predicting from regression took ', str(time.time() - t))

        return Y_pred


class XarrayCovariateRegression:
    """
    Adds alignment-checking, un- and re-packaging, and comparison functionality to a regression.
    """

    def __init__(self, regression, expected_dims=Defaults.expected_dims, neuroid_dim=Defaults.neuroid_dim,
                 neuroid_coord=Defaults.neuroid_coord, stimulus_coord=Defaults.stimulus_coord):
        self._regression = regression
        self._expected_dims = expected_dims
        self._neuroid_dim = neuroid_dim
        self._neuroid_coord = neuroid_coord
        self._stimulus_coord = stimulus_coord
        self._target_neuroid_values = None

    def fit(self, source, covariate, target):
        source, covariate, target = self._align(source), self._align(covariate), self._align(target)
        source, covariate, target = source.sortby(self._stimulus_coord), covariate.sortby(
            self._stimulus_coord), target.sortby(self._stimulus_coord)

        self._regression.fit(source, covariate, target)

        self._target_neuroid_values = {}
        for name, dims, values in walk_coords(target):
            if self._neuroid_dim in dims:
                assert array_is_element(dims, self._neuroid_dim)
                self._target_neuroid_values[name] = values

    def predict(self, source, covariate):
        source, covariate = self._align(source), self._align(covariate)
        source, covariate = source.sortby(self._stimulus_coord), covariate.sortby(self._stimulus_coord)
        predicted_values = self._regression.predict(source, covariate)
        prediction = self._package_prediction(predicted_values, source=source)
        return prediction

    def _package_prediction(self, predicted_values, source):
        coords = {coord: (dims, values) for coord, dims, values in walk_coords(source)
                  if not array_is_element(dims, self._neuroid_dim)}
        # re-package neuroid coords
        dims = source.dims
        # if there is only one neuroid coordinate, it would get discarded and the dimension would be used as coordinate.
        # to avoid this, we can build the assembly first and then stack on the neuroid dimension.
        neuroid_level_dim = None
        if len(self._target_neuroid_values) == 1:  # extract single key: https://stackoverflow.com/a/20145927/2225200
            (neuroid_level_dim, _), = self._target_neuroid_values.items()
            dims = [dim if dim != self._neuroid_dim else neuroid_level_dim for dim in dims]
        for target_coord, target_value in self._target_neuroid_values.items():
            # this might overwrite values which is okay
            coords[target_coord] = (neuroid_level_dim or self._neuroid_dim), target_value
        prediction = NeuroidAssembly(predicted_values, coords=coords, dims=dims)
        if neuroid_level_dim:
            prediction = prediction.stack(**{self._neuroid_dim: [neuroid_level_dim]})

        return prediction

    def _align(self, assembly):
        assert set(assembly.dims) == set(self._expected_dims), \
            f"Expected {set(self._expected_dims)}, but got {set(assembly.dims)}"
        return assembly.transpose(*self._expected_dims)


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

    def _unflatten(self, X, channel_coord=None, image_dir = None):
        """
        Unflattens NeuroidAssembly of flattened model activations to
        BxCxH*W (or BXCxW*H not sure, also not sure if it matters)

        Using the information in coordinates channel, channel_x, channel_y which give the index along each
        of the original axes

        Order of coordinates (which one represents first axis, second, third) is determined by checking which one's
        values change slowest (i.e., reverse sort by first occurence of a 1-value)
        """
        n_viz = 10

        # Get first n_viz image paths for visualizations
        if image_dir:
            vis_images_fnames = X.image_file_name.values[0:n_viz]
            vis_images_fpaths = [os.path.join(image_dir, file_name) for file_name in vis_images_fnames]

            dest_dir = "actvns_viz"
            os.makedirs(dest_dir, exist_ok=True)
            dest_fname = '_'.join([str(self.__class__.__name__), X.model.values[0], X.layer.values[0]]) + '.png'

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

        # Channels first
        channel_index = [i for i, (key, value) in enumerate(X_shape.items()) if key==channel_coord][0]
        channel_index = channel_index + 1 # bc very first is B
        transpose_to = [0, channel_index]+ [i for i in [1,2,3] if i != channel_index]
        X = np.transpose(X, transpose_to)

        # Make visualizations
        if image_dir:
            actvns = X[0:n_viz, :,:,:].mean(axis=1)
            actvns = np.split(actvns, n_viz, axis=0)
            actvns = [np.squeeze(actvn) for actvn in actvns]

            ims = [np.array(Image.open(fpath)) for fpath in vis_images_fpaths]

            ims_actvns = [val for pair in zip(ims, actvns) for val in pair]

            rows = n_viz
            cols = 2
            axes = []
            fig = plt.figure(figsize=(2,10))

            for i in range(rows * cols):
                b = ims_actvns[i]
                axes.append(fig.add_subplot(rows, cols, i + 1))
                plt.imshow(b)

            fig.set_figheight(50)
            fig.set_figwidth(50)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()
            plt.savefig(os.path.join(dest_dir, dest_fname))

        # reshape to BxCxH*W (or W*H, not sure)
        X = X.reshape(list(X.shape[0:2])+[-1])

        return X

    def _preprocess_gram(self, X, fit=True, image_dir=None):
        # Center/scale
        if fit:
            self.scaler_x.fit(X)
        X.values = self.scaler_x.transform(X)

        # Compute gram matrices
        X_grams = self._unflatten(X, self.channel_coord, image_dir=image_dir) # Unflatten X to BxCxH*W (or W*H, not sure)
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
            X = self._preprocess_gram(X, fit=True, image_dir=os.path.dirname(Y.stimulus_set.get_image(Y.image_id.values[0])))
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


class OldGramControlPLS():
    def __init__(self, gram_control=False, channel_coord=None, regression_kwargs=None):
        self.gram_control = gram_control
        self.channel_coord = channel_coord
        self.regression_kwargs = regression_kwargs or {}
        self.control_regression = PLSRegression(**self.regression_kwargs)
        self.main_regression = PLSRegression(**self.regression_kwargs)

    def _unflatten(self, X, channel_coord=None, image_dir=None):
        """
        Unflattens NeuroidAssembly of flattened model activations to
        BxCxH*W (or BXCxW*H not sure, also not sure if it matters)

        Using the information in coordinates channel, channel_x, channel_y which give the index along each
        of the original axes

        Order of coordinates (which one represents first axis, second, third) is determined by checking which one's
        values change slowest (i.e., reverse sort by first occurence of a 1-value)
        """

        n_viz = 10

        # Get first n_viz image paths for visualizations
        if image_dir:
            vis_images_fnames = X.image_file_name.values[0:n_viz]
            vis_images_fpaths = [os.path.join(image_dir, file_name) for file_name in vis_images_fnames]

            dest_dir = "actvns_viz"
            os.makedirs(dest_dir, exist_ok=True)
            dest_fname = '_'.join([str(self.__class__.__name__), X.model.values[0], X.layer.values[0]]) + '.png'


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

        # Channels first
        channel_index = [i for i, (key, value) in enumerate(X_shape.items()) if key==channel_coord][0]
        channel_index = channel_index + 1 # bc very first is B
        transpose_to = [0, channel_index]+ [i for i in [1,2,3] if i != channel_index]
        X = np.transpose(X, transpose_to)

        # Make visualizations
        if image_dir:
            actvns = X[0:n_viz, :, :, :].mean(axis=1)
            actvns = np.split(actvns, n_viz, axis=0)
            actvns = [np.squeeze(actvn) for actvn in actvns]

            ims = [np.array(Image.open(fpath)) for fpath in vis_images_fpaths]

            ims_actvns = [val for pair in zip(ims, actvns) for val in pair]

            rows = n_viz
            cols = 2
            axes = []
            fig = plt.figure(figsize=(2, 10))

            for i in range(rows * cols):
                b = ims_actvns[i]
                axes.append(fig.add_subplot(rows, cols, i + 1))
                plt.imshow(b)

            fig.set_figheight(50)
            fig.set_figwidth(50)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()
            plt.savefig(os.path.join(dest_dir, dest_fname))

        # Reshape to BxCxH*W (or W*H, not sure)
        X = X.reshape(list(X.shape[0:2])+[-1])

        return X

    def _preprocess_gram(self, X, fit=True, image_dir=None):

        # Compute gram matrices
        X_grams = self._unflatten(X, self.channel_coord, image_dir) # Unflatten X to BxCxH*W (or W*H, not sure)
        X_grams = np.einsum("ijk, ikl -> ijl", X_grams, np.transpose(X_grams, [0,2,1]))
        #X_grams = X_grams/X.size # is this the right normalization?
        X_grams = X_grams.reshape(X_grams.shape[0], -1)

        # Residuals
        if fit:
            self.control_regression.fit(X_grams, X)
        X = X - self.control_regression.predict(X_grams)

        return X

    def _preprocess(self, X, fit=True):
        # I think all the preprocessing needed happens inside PLS?
        return X


    def fit(self, X, Y):
        if self.gram_control:
            X = self._preprocess_gram(X, fit=True, image_dir=os.path.dirname(Y.stimulus_set.get_image(Y.image_id.values[0])))
        else:
            X = self._preprocess(X, fit=True)

        self.main_regression.fit(X, Y)

    def predict(self, X):
        if self.gram_control:
            X = self._preprocess_gram(X, fit=False)
        else:
            X = self._preprocess(X, fit=False)

        Ypred = self.main_regression.predict(X)

        return Ypred


def old_gram_control_regression(gram_control, channel_coord=None, scaler_kwargs=None, pca_kwargs=None, regression_kwargs=None, xarray_kwargs=None):
    scaler_defaults = dict(with_std=False)
    pca_defaults = dict(n_components=25)
    scaler_kwargs = {**scaler_defaults, **(scaler_kwargs or {})}
    pca_kwargs = {**pca_defaults, **(pca_kwargs or {})}
    regression_kwargs = regression_kwargs or {}
    regression = GramControlRegression(gram_control=gram_control,
                                       channel_coord=channel_coord,
                                       scaler_kwargs=scaler_kwargs,
                                       pca_kwargs=pca_kwargs,
                                       regression_kwargs=regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression

def old_gram_control_pls(gram_control, channel_coord=None, regression_kwargs=None, xarray_kwargs=None):
    regression_defaults = dict(n_components=25, scale=False)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression = OldGramControlPLS(gram_control=gram_control, channel_coord=channel_coord, regression_kwargs=regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def semipartial_regression(covariate_control=False, scaler_kwargs=None, pca_kwargs=None, regression_kwargs=None,
                           xarray_kwargs=None):
    scaler_defaults = dict(with_std=False)
    pca_defaults = dict(n_components=25)
    scaler_kwargs = {**scaler_defaults, **(scaler_kwargs or {})}
    pca_kwargs = {**pca_defaults, **(pca_kwargs or {})}
    regression_kwargs = regression_kwargs or {}
    regression = SemiPartialRegression(covariate_control=covariate_control,
                                       scaler_kwargs=scaler_kwargs,
                                       pca_kwargs=pca_kwargs,
                                       regression_kwargs=regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayCovariateRegression(regression, **xarray_kwargs)
    return regression


def semipartial_pls(covariate_control=False, regression_kwargs=None, xarray_kwargs=None):
    regression_defaults = dict(n_components=25, scale=False)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression = SemiPartialPLS(covariate_control=covariate_control, regression_kwargs=regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayCovariateRegression(regression, **xarray_kwargs)
    return regression


def gram_pls(regression_kwargs=None, xarray_kwargs=None):
    regression_defaults = dict(n_components=25, scale=False)
    regression_kwargs = {**regression_defaults, **(regression_kwargs or {})}
    regression = GramPLS(regression_kwargs=regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def gram_linear(gram=True, pca_treshold=None, scaler_kwargs=None, pca_kwargs=None, regression_kwargs=None, xarray_kwargs=None):
    scaler_defaults = dict(with_std=False)
    pca_defaults = dict(n_components=None)  # instead of 25 because we are worried about how much variance is explained
    scaler_kwargs = {**scaler_defaults, **(scaler_kwargs or {})}
    pca_kwargs = {**pca_defaults, **(pca_kwargs or {})}
    regression_kwargs = regression_kwargs or {}
    regression = GramLinearRegression(gram=gram,
                                      channel_coord=None,
                                      scaler_kwargs=scaler_kwargs,
                                      pca_kwargs=pca_kwargs,
                                      regression_kwargs=regression_kwargs)
    xarray_kwargs = xarray_kwargs or {}
    regression = XarrayRegression(regression, **xarray_kwargs)
    return regression


def unflatten(X, channel_coord=None, image_dir=None):
    """
    Unflattens NeuroidAssembly of flattened model activations to
    BxCxH*W (or BXCxW*H not sure, also not sure if it matters)

    Using the information in coordinates channel, channel_x, channel_y which give the index along each
    of the original axes

    Order of coordinates (which one represents first axis, second, third) is determined by checking which one's
    values change slowest (i.e., reverse sort by first occurence of a 1-value)
    """

    n_viz = 10

    # Get first n_viz image paths for visualizations
    if image_dir:
        vis_images_fnames = X.image_file_name.values[0:n_viz]
        vis_images_fpaths = [os.path.join(image_dir, file_name) for file_name in vis_images_fnames]

        dest_dir = "actvns_viz"
        os.makedirs(dest_dir, exist_ok=True)
        dest_fname = 'activations.png'

    # Get coord:(original axis length, first occurence of 1
    X_shape = OrderedDict({
        'channel': (X.channel.values.max() + 1, np.where(X.channel.values == 1)[0][0]),
        'channel_x': (X.channel_x.values.max() + 1, np.where(X.channel_x.values == 1)[0][0]),
        'channel_y': (X.channel_y.values.max() + 1, np.where(X.channel_y.values == 1)[0][0])
    })

    # Determine which coordinate represents channels
    if channel_coord == None:
        # Hacky attempt to determine automatically (usually W==H != C)
        X_axis_sizes = [i[0] for i in X_shape.values()]
        frequencies = {key: Counter(X_axis_sizes)[value[0]] for key, value in X_shape.items()}
        if sorted(list(frequencies.values())) != [1, 2, 2]:
            raise ValueError('channel_coord is None and failed to automatically determine it')
        else:
            channel_coord = [key for key, value in frequencies.items() if value == 1][0]

    # Sort coordinates such that first one represents first axis in original matrix, etc.
    X_shape = OrderedDict(sorted(X_shape.items(), key=lambda x: x[1][1], reverse=True))

    # Unflatten X
    B = X.shape[0]
    reshape_to = [B] + [value[0] for key, value in X_shape.items()]
    X = X.values.reshape(reshape_to)

    # Channels first
    channel_index = [i for i, (key, value) in enumerate(X_shape.items()) if key == channel_coord][0]
    channel_index = channel_index + 1  # bc very first is B
    transpose_to = [0, channel_index] + [i for i in [1, 2, 3] if i != channel_index]
    X = np.transpose(X, transpose_to)

    # Make visualizations
    if image_dir:
        actvns = X[0:n_viz, :, :, :].mean(axis=1)
        actvns = np.split(actvns, n_viz, axis=0)
        actvns = [np.squeeze(actvn) for actvn in actvns]

        ims = [np.array(Image.open(fpath)) for fpath in vis_images_fpaths]

        ims_actvns = [val for pair in zip(ims, actvns) for val in pair]

        rows = n_viz
        cols = 2
        axes = []
        fig = plt.figure(figsize=(2, 10))

        for i in range(rows * cols):
            b = ims_actvns[i]
            axes.append(fig.add_subplot(rows, cols, i + 1))
            plt.imshow(b)

        fig.set_figheight(50)
        fig.set_figwidth(50)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        plt.savefig(os.path.join(dest_dir, dest_fname))

    return X


def take_gram(X):
    """
    X needs to be of shape:
    BxCxH*W (or BXCxW*H not sure, also not sure if it matters)

    Computes the gram matrix for each of B samples and flattens

    """
    X_grams = einsum("ijk, ikl -> ijl", X, np.transpose(X, [0, 2, 1]))
    # X_grams = X_grams/X.size # is this the right normalization?
    C = X_grams.shape[1]
    X_grams = X_grams.reshape(X_grams.shape[0], -1)
    X_grams = X_grams[:, np.ravel_multi_index(np.triu_indices(C), dims=(C,C))]

    return X_grams
