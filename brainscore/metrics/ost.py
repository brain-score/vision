from collections import OrderedDict

import numpy as np
import sklearn.linear_model
import sklearn.multioutput
from scipy.stats import pearsonr

from brainio_base.assemblies import walk_coords, array_is_element, BehavioralAssembly
from brainscore.metrics import Metric
from brainscore.metrics.behavior import I1
from brainscore.metrics.transformations import CrossValidation


class OSTCorrelation(Metric):
    def __init__(self):
        self._cross_validation = CrossValidation(stratification_coord=None)
        self._i1 = I1()

    def __call__(self, source_recordings, target_osts):
        score = self._cross_validation(source_recordings, target_osts, apply=self.apply)
        return score

    def apply(self, train_source, train_osts, test_source, test_osts):
        predicted_osts = self.compute_osts(train_source, test_source, test_osts)
        score = self.correlate(predicted_osts, test_osts)
        return score

    def compute_osts(self, train_source, test_source, test_osts):
        predicted_osts = np.empty(len(test_osts), dtype=np.float)
        predicted_osts[:] = np.nan
        for time_bin in sorted(set(train_source['time_bin'].values)):
            time_train_source = train_source.sel(time_bin=time_bin)
            time_test_source = test_source.sel(time_bin=time_bin)
            time_train_source = time_train_source.transpose('presentation', 'neuroid')
            time_test_source = time_test_source.transpose('presentation', 'neuroid')
            classifier = ProbabilitiesClassifier()
            classifier.fit(time_train_source, time_train_source['image_label'])
            prediction_probabilities = classifier.predict_proba(time_test_source)
            source_i1 = self.i1(prediction_probabilities)
            assert all(source_i1['image_id'].values == test_osts['image_id'].values)
            for i, (image_source_i1, threshold_i1) in enumerate(zip(
                    source_i1.values, test_osts['i1'].values)):
                if np.isnan(predicted_osts[i]) and image_source_i1 >= threshold_i1:
                    predicted_osts[i] = np.mean(time_bin)  # TODO: extrapolate
            if not any(np.isnan(predicted_osts)):
                break
        return predicted_osts

    def correlate(self, predicted_osts, target_osts):
        correlation, p = pearsonr(predicted_osts, target_osts)
        return correlation

    def i1(self, prediction_probabilities):
        response_matrix = self._i1.target_distractor_scores(prediction_probabilities)
        response_matrix = self._i1.dprimes(response_matrix)
        response_matrix = self._i1.collapse_distractors(response_matrix)
        return response_matrix


class ProbabilitiesClassifier:
    def __init__(self, classifier_c=1e-3):
        self._classifier = sklearn.linear_model.LogisticRegression(
            multi_class='multinomial', solver='newton-cg', C=classifier_c)
        self._label_mapping = None
        self._scaler = None

    def fit(self, X, Y):
        self._scaler = sklearn.preprocessing.StandardScaler().fit(X)
        X = self._scaler.transform(X)
        Y, self._label_mapping = self.labels_to_indices(Y.values)
        self._classifier.fit(X, Y)
        return self

    def predict_proba(self, X):
        assert len(X.shape) == 2, "expected 2-dimensional input"
        scaled_X = self._scaler.transform(X)
        proba = self._classifier.predict_proba(scaled_X)
        # we take only the 0th dimension because the 1st dimension is just the features
        X_coords = {coord: (dims, value) for coord, dims, value in walk_coords(X)
                    if array_is_element(dims, X.dims[0])}
        proba = BehavioralAssembly(proba,
                                   coords={**X_coords, **{'choice': list(self._label_mapping.values())}},
                                   dims=[X.dims[0], 'choice'])
        return proba

    def labels_to_indices(self, labels):
        label2index = OrderedDict()
        indices = []
        for label in labels:
            if label not in label2index:
                label2index[label] = (max(label2index.values()) + 1) if len(label2index) > 0 else 0
            indices.append(label2index[label])
        index2label = OrderedDict((index, label) for label, index in label2index.items())
        return indices, index2label
