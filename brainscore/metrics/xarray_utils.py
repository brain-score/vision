import numpy as np

from brainscore.assemblies import NeuroidAssembly, DataAssembly
from brainscore.assemblies import array_is_element, walk_coords


class Defaults:
    expected_dims = ('presentation', 'neuroid')
    stimulus_coord = 'image_id'
    neuroid_dim = 'neuroid'
    neuroid_coord = 'neuroid_id'


class XarrayRegression:
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

    def fit(self, source, target):
        np.testing.assert_array_equal(source.dims, self._expected_dims)
        np.testing.assert_array_equal(target.dims, self._expected_dims)
        assert all(source[self._stimulus_coord].values == target[self._stimulus_coord].values)

        self._regression.fit(source, target)

        self._target_neuroid_values = {}
        for name, dims, values in walk_coords(target):
            if self._neuroid_dim in dims:
                assert array_is_element(dims, self._neuroid_dim)
                self._target_neuroid_values[name] = values

    def predict(self, source):
        np.testing.assert_array_equal(source.dims, self._expected_dims)

        predicted_values = self._regression.predict(source)

        prediction = self.package_prediction(predicted_values, source=source)
        return prediction

    def package_prediction(self, predicted_values, source):
        coords = {coord: (dims, values) for coord, dims, values in walk_coords(source)
                  if not array_is_element(dims, self._neuroid_dim)}
        # re-package neuroid coords
        for target_coord, target_value in self._target_neuroid_values.items():
            coords[target_coord] = self._neuroid_dim, target_value  # this might overwrite values which is okay
        prediction = NeuroidAssembly(predicted_values, coords=coords, dims=source.dims)
        return prediction


class XarrayCorrelation:
    def __init__(self, correlation, stimulus_coord=Defaults.stimulus_coord, neuroid_coord=Defaults.neuroid_coord):
        self._correlation = correlation
        self._stimulus_coord = stimulus_coord
        self._neuroid_coord = neuroid_coord

    def __call__(self, prediction, target):
        assert all(prediction[self._stimulus_coord].values == target[self._stimulus_coord].values)
        assert all(prediction[self._neuroid_coord].values == target[self._neuroid_coord].values)
        correlations = []
        for i in target[self._neuroid_coord].values:
            target_activations = target.sel(**{self._neuroid_coord: i}).squeeze()
            prediction_activations = prediction.sel(**{self._neuroid_coord: i}).squeeze()
            r, p = self._correlation(target_activations, prediction_activations)
            correlations.append(r)

        neuroid_dim = target[self._neuroid_coord].dims
        result = DataAssembly(correlations,
                              coords={coord: (dims, values)
                                      for coord, dims, values in walk_coords(target) if dims == neuroid_dim},
                              dims=neuroid_dim)
        return result
