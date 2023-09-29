
import numpy as np
from sklearn.linear_model import LinearRegression
from brainscore.metrics import Score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.stats import pearsonr as corr

class RegressedCorrelationAlgonauts:
    ''' 
    This class performs regression-based correlation analysis on brain imaging data.
    It takes source and target data, preprocesses the data, applies regression,
    and calculates correlations.
    The train and test split, as well as the ceiling, are calculated 
    accordingly to algonauts 2023 challenge.
    '''
    def __init__(self, regression, correlation):
        regression = regression or linear_regression

        self.regression = regression
        self.correlation = correlation
    
    def __call__(self, source, target):
        
            # =============================================================================
            # Extracting the features from the ANN
            # =============================================================================
            print('initial source shape: ', source.shape)
            print('initial target shape: ', target.shape)

            print('source.stimulus_id: ', source.stimulus_id.data)
            print('target.stimulus_id: ', target.stimulus_id.data)
        
            # divide into train and test
            train_idx = np.where(target.algonauts_train_test == 'train')[0]
            test_idx = np.where(target.algonauts_train_test == 'test')[0]

            source_train = source[train_idx].data
            target_train = target[train_idx].data
            print('source_train shape: ', source_train.shape)
            print('target_train shape: ', target_train.shape)


            source_test = source[test_idx].data
            target_test = target[test_idx].data
            print('source_test shape: ', source_test.shape)
            print('target_test shape: ', target_test.shape)

            # Set random seed for reproducible results (algonauts seed)
            seed = 20200220 

            #Standardize the data
            print('Standardization of the features')
            sc = StandardScaler()
            sc.fit(source_train)
            source_train = sc.transform(source_train)

            # Apply PCA to train test images features maps
            print('Fitting PCA')
            pca = PCA(n_components=100, random_state=seed)
            pca.fit(source_train)
            print('Transforming TRAIN features using PCA')
            source_train = pca.transform(source_train)

            # Standardizing and applying the PCA to TEST as well
            print('Transforming TEST features')
            source_test = sc.transform(source_test)
            source_test = pca.transform(source_test)
            
            # =============================================================================
            # TRAIN THE LINEAR REGRESSION ENCODING & Get PREDICTIONS
            # =============================================================================
            
            target_train_lh = target.sel(hemisphere = 'left')[train_idx].data
            target_train_rh = target.sel(hemisphere = 'right')[train_idx].data
            
            # Train the linear regression and save the predicted data
            # Empty synthetic fMRI data matrices of shape:
            # (Test image conditions × fMRI vertices)
            lh_pred_test = np.zeros((source_test.shape[0],target_train_lh.shape[1]))
            rh_pred_test = np.zeros((source_test.shape[0],target_train_rh.shape[1]))

            # Independently for each fMRI vertex, fit a linear regressiosn using the
            # training image conditions and use it to synthesize the fMRI responses of the
            # test image conditions
            for v in tqdm(range(target_train_lh.shape[1]), desc='Linear Regression LH'):
                reg_lh = LinearRegression().fit(source_train, target_train_lh[:,v])
                lh_pred_test[:,v] = reg_lh.predict(source_test)
            for v in tqdm(range(target_train_rh.shape[1]), desc='Linear Regression RH'):
                reg_rh = LinearRegression().fit(source_train, target_train_rh[:,v])
                rh_pred_test[:,v] = reg_rh.predict(source_test)


            # =============================================================================
            # Evaluate the Encoding (Raw correlation, no Noise Normalization)
            # =============================================================================
            target_test_lh = target.sel(hemisphere = 'left')[test_idx].data
            target_test_rh = target.sel(hemisphere = 'right')[test_idx].data

            # Left hemishpere
            lh_correlation = np.zeros(target_test_lh.shape[1])
            for v in range(target_test_lh.shape[1]):
                lh_correlation[v] = corr(target_test_lh[:,v], lh_pred_test[:,v])[0]

            # Right hemishpere
            rh_correlation = np.zeros(target_test_rh.shape[1])
            for v in range(target_test_rh.shape[1]):
                rh_correlation[v] = corr(target_test_rh[:,v], rh_pred_test[:,v])[0]

            
            correlations = np.concatenate((lh_correlation, rh_correlation))
            center = correlations
            # error =  np.std(correlations) / np.sqrt(len(correlations))
            error = 0
            score =  Score([center, error],
                        coords={**{'aggregation': ['center', 'error']}},
                        dims=('aggregation',))
            # score.attrs['raw'] = correlations
            return score
            

def linear_regression(xarray_kwargs=None):
    regression = LinearRegression()
    return regression

def pearsonr_correlation(xarray_kwargs=None):
    return corr

