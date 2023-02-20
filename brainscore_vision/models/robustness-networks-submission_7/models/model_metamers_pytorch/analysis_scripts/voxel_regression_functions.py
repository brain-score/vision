"""
Based on the fMRI scripts used in Kell et al. 2018 with minor changes. 
"""

import h5py
import numpy as np
import pickle

import sklearn
from sklearn import *
import scipy

import os

def compute_median_across_predictions(combine_predictions):
    """
    Computes a summary median metric from a set of predictions collected
    across multiple layers of a network.
    
    Input:
        combine_predictions (numpy array): array containing corrected 
            R^2 values in shape of [layers, voxels, splits]. 

    Output: 
        median_from_best_layers (float): a value representing the median best prediction 
           value, by taking the layer with the best prediction for each voxel. 
        std_across_splits (float): the standard deviation of the median voxel predictions
    """
    # median across splits of the data
    median_across_splits = np.nanmedian(combine_predictions,2)
    # get the best layer for each of the voxels
    arg_layer_max = np.nanargmax(median_across_splits, 0)
    # take median across the best layer
    num_voxels = combine_predictions.shape[1]
    median_from_best_layers = np.median(median_across_splits[arg_layer_max, np.arange(num_voxels)])

    # take the median across voxels, choosing the best layer from above, and take the STD across splits
    median_across_voxels = np.median(combine_predictions[arg_layer_max, np.arange(num_voxels), :], 0)
    std_across_splits = np.std(median_across_voxels)

    return median_from_best_layers, std_across_splits

def compute_best_median_for_each_voxel(combine_predictions):
    """
    Computes the median of the best predictions layer for each voxel

    Input:
        combine_predictions (numpy array): array containing corrected
            R^2 values in shape of [layers, voxels, splits].

    Output:
        best_median_for_voxels (np array): the median from the best predicting layer for each
            voxel
        voxel_std_across_splits (np array): the standard deviation of each median voxel prediction
    """
    # median across splits of the data
    median_across_splits = np.nanmedian(combine_predictions,2)
    # get the best layer for each of the voxels
    arg_layer_max = np.nanargmax(median_across_splits, 0)
    # take median across the best layer
    num_voxels = combine_predictions.shape[1]
    best_median_for_voxels = median_across_splits[arg_layer_max, np.arange(num_voxels)]

    # take the median across voxels, choosing the best layer from above, and take the STD across splits
    median_across_voxels = combine_predictions[arg_layer_max, np.arange(num_voxels), :]
    voxel_std_across_splits = np.std(median_across_voxels, 1)

    return best_median_for_voxels, voxel_std_across_splits

# Need to define the following
def ridgeRegressSplits(features,
                       y,
                       is_train_data,
                       is_test_data,
                       possible_alphas,
                       voxel_idx=None, 
                       ridge_normalize=False,
                       ridge_fit_intercept=True, 
                       do_return_y_hats=True,
                       do_suppress_stdout=True,
                       do_return_coeffs=True,
                       zero_center=False):

    """
    Performs ridge regression, using a specified set of x and y

    Returns the uncorrected r^2, alpha, and yhat

    """
   
    ridgeCrossVal = sklearn.linear_model.RidgeCV(alphas=possible_alphas, normalize=ridge_normalize, fit_intercept=ridge_fit_intercept)
    X_train = features[is_train_data,:]
    Y_train = y[is_train_data]

    if zero_center:
        scaler = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=False)
        scaler.fit(Y_train.reshape(-1,1))
        Y_train = scaler.transform(Y_train.reshape(-1,1)).ravel()

    ridgeCrossVal.fit(X_train, Y_train)
        
    X_test = features[is_test_data,:]
    Y_test = y[is_test_data]
    if zero_center:
        Y_test = scaler.transform(Y_test.reshape(-1,1)).ravel()

    alpha = ridgeCrossVal.alpha_
    
    if (alpha==possible_alphas[0]) or (alpha==possible_alphas[-1]):
        print('WARNING: BEST ALPHA %.2E IS AT THE EDGE OF THE SEARCH SPACE FOR VOXEL %d'%(alpha, voxel_idx))
    
    yhat = ridgeCrossVal.predict(X_test)

    if not np.isclose(np.std(yhat),0):
        try: 
            r2, p = scipy.stats.pearsonr(yhat.ravel(), Y_test.ravel())
        except(AttributeError): 
            print(yhat.ravel())
            print(Y_test.ravel())
    else: 
        print('WARNING: VOXEL %d is predicted as only the expected value. Setting correlation to zero.'%voxel_idx)
        r2 = 0
    r2 = r2**2
    
    return r2, alpha, yhat


def runRidgeWithCorrectedR2_ThreeRunSplit(features, 
                                          voxel_data, 
                                          voxel_idx, 
                                          is_train_data, 
                                          is_test_data, 
                                          possible_alphas,
                                          zero_center=False):

    r2_mean, alpha_mean, yhat_mean = ridgeRegressSplits(features,
                           np.mean(voxel_data,2)[:,voxel_idx][:,None],
                           is_train_data,
                           is_test_data,
                           possible_alphas,
                           voxel_idx=voxel_idx, 
                           do_return_y_hats=True,
                           do_suppress_stdout=True,
                           do_return_coeffs=True,
                           zero_center=zero_center)
        
    split_rs = []	
    split_alphas = []
    split_yhat = []

    for run in [0,1,2]: 
        r2, alpha, yhat = ridgeRegressSplits(features,
                           voxel_data[:,:,run][:,voxel_idx],
                           is_train_data,
                           is_test_data,
                           possible_alphas,
                           voxel_idx = voxel_idx, 
                           do_return_y_hats=True,
                           do_suppress_stdout=True,
                           do_return_coeffs=True,
                           zero_center=zero_center)

        if not np.isnan(r2): 
           split_rs.append(np.sqrt(r2)) # only want the r here for the splits. 
        else:
           print('WARNING: VOXEL %d contains nans in the r2'%voxel_idx)
           split_rs.append(r2)
        split_alphas.append(alpha)
        split_yhat.append(yhat)

    rvs = []
    rv_hats = []

    for split_idx, split in enumerate([[0,1], [1,2], [0,2]]):
        split_r, split_p = scipy.stats.pearsonr(voxel_data[is_train_data,voxel_idx,split[0]],
                                                voxel_data[is_train_data,voxel_idx,split[1]])
        rvs.append(split_r)

        if (not np.isclose(np.std(split_yhat[split[0]]),0)) and (not np.isclose(np.std(split_yhat[split[1]]),0)):
            split_r_hat, split_p_hat = scipy.stats.pearsonr(split_yhat[split[0]],split_yhat[split[1]])
        else: # if one of the predictions is constant, then set the r^2 to zero. 
            print('WARNING: VOXEL %d, Run %s is predicted as only the expected value. Setting correlation to zero.'%(voxel_idx, split))
            split_r_hat = 0
        rv_hats.append(split_r_hat)

    # Note that for very noisy voxels, this division by the estimated reliability can be unstable 
    # and can cause for corrected variance explained measures that exceed one. To ameliorate this 
    # problem, we limited both the reliability of the prediction and the reliability of the voxel 
    # response to be greater than some value k (Huth et al., 2016). 
    rv_med = np.nanmedian(rvs)
    rv = max(3*rv_med/(1+2*rv_med),  0.182) # Value changed to 0.182 which corresponds to p<0.05 for correlations of two random gaussian variables of length 83 (length of the training data)
    rv_hat_med = np.nanmedian(rv_hats)
    rv_hat = max(3*rv_hat_med/(1+2*rv_hat_med), 0.183) # Value changed to 0.183 which corresponds to p<0.05 for correlations of two random gaussian variables of length 82 (length of the test data)

    corrected_r2 = r2_mean/(rv*rv_hat)
    
    return corrected_r2, alpha_mean
