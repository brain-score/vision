import numpy as np
from sklearn.linear_model import LinearRegression
from brainscore.metrics import Score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr as corr

class RegressedCorrelationAlgonauts:
    ''' 
    This class performs regression-based correlation analysis on brain imaging data.
    It takes source and target data, preprocesses the data, applies regression,
    and calculates correlations.
    The train and test split, as well as the ceiling, are calculated 
    accordingly to Algonauts 2023 challenge.
    '''
    def __init__(self, regression, correlation):
        regression = regression or linear_regression

        self.regression = regression
        self.correlation = correlation
    
    def __call__(self, source, target):
        # Extracting the features from the ANN
        # Divide into train and test
        train_idx = np.where(target.algonauts_train_test == 'train')[0]
        test_idx = np.where(target.algonauts_train_test == 'test')[0]

        source_train = source[train_idx].data
        target_train = target[train_idx].data

        source_test = source[test_idx].data
        target_test = target[test_idx].data

        # Set a random seed for reproducible results (Algonauts seed)
        seed = 20200220 

        # Standardize the data
        sc = StandardScaler()
        sc.fit(source_train)
        source_train = sc.transform(source_train)

        # Apply PCA to train test images features maps
        pca = PCA(n_components=100, random_state=seed)
        pca.fit(source_train)
        source_train = pca.transform(source_train)

        # Standardizing and applying PCA to TEST as well
        source_test = sc.transform(source_test)
        source_test = pca.transform(source_test)

        # TRAIN THE LINEAR REGRESSION ENCODING & Get PREDICTIONS

        target_train_lh = target.sel(hemisphere='left')[train_idx].data
        target_train_rh = target.sel(hemisphere='right')[train_idx].data

        # Train the linear regression and save the predicted data
        # Empty synthetic fMRI data matrices of shape:
        # (Test image conditions × fMRI vertices)
        lh_pred_test = np.zeros((source_test.shape[0], target_train_lh.shape[1]))
        rh_pred_test = np.zeros((source_test.shape[0], target_train_rh.shape[1]))

        # Independently for each fMRI vertex, fit a linear regression using the
        # training image conditions and use it to synthesize the fMRI responses of the
        # test image conditions
        for v in range(target_train_lh.shape[1]):
            reg_lh = LinearRegression().fit(source_train, target_train_lh[:, v])
            lh_pred_test[:, v] = reg_lh.predict(source_test)
        for v in range(target_train_rh.shape[1]):
            reg_rh = LinearRegression().fit(source_train, target_train_rh[:, v])
            rh_pred_test[:, v] = reg_rh.predict(source_test)

        # Evaluate the Encoding (Raw correlation, no Noise Normalization)

        target_test_lh = target.sel(hemisphere='left')[test_idx].data
        target_test_rh = target.sel(hemisphere='right')[test_idx].data

        # Left hemisphere
        lh_correlation = np.zeros(target_test_lh.shape[1])
        for v in range(target_test_lh.shape[1]):
            lh_correlation[v] = corr(target_test_lh[:, v], lh_pred_test[:, v])[0]

        # Right hemisphere
        rh_correlation = np.zeros(target_test_rh.shape[1])
        for v in range(target_test_rh.shape[1]):
            rh_correlation[v] = corr(target_test_rh[:, v], rh_pred_test[:, v])[0]

        correlations = np.concatenate((lh_correlation, rh_correlation))
        score = Score(correlations)
        return score

def linear_regression(xarray_kwargs=None):
    regression = LinearRegression()
    return regression

def pearsonr_correlation(xarray_kwargs=None):
    return corr