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
        source, target = self._align(source), self._align(target)
        source, target = source.sortby(self._stimulus_coord), target.sortby(self._stimulus_coord)

        self._regression.fit(source, target)

        self._target_neuroid_values = {}
        for name, dims, values in walk_coords(target):
            if self._neuroid_dim in dims:
                assert array_is_element(dims, self._neuroid_dim)
                self._target_neuroid_values[name] = values

    def predict(self, source):
        source = self._align(source)

        predicted_values = self._regression.predict(source)

        prediction = self._package_prediction(predicted_values, source=source)
        return prediction

    def _package_prediction(self, predicted_values, source):
        coords = {coord: (dims, values) for coord, dims, values in walk_coords(source)
                  if not array_is_element(dims, self._neuroid_dim)}
        # re-package neuroid coords
        for target_coord, target_value in self._target_neuroid_values.items():
            coords[target_coord] = self._neuroid_dim, target_value  # this might overwrite values which is okay
        prediction = NeuroidAssembly(predicted_values, coords=coords, dims=source.dims)
        return prediction

    def _align(self, assembly):
        assert set(assembly.dims) == set(self._expected_dims)
        return assembly.transpose(*self._expected_dims)


class XarrayCorrelation:
    def __init__(self, correlation, stimulus_coord=Defaults.stimulus_coord, neuroid_coord=Defaults.neuroid_coord):
        self._correlation = correlation
        self._stimulus_coord = stimulus_coord
        self._neuroid_coord = neuroid_coord

    def __call__(self, prediction, target):
        # align
        prediction = prediction.sortby([self._stimulus_coord, self._neuroid_coord])
        target = target.sortby([self._stimulus_coord, self._neuroid_coord])
        assert np.array(prediction[self._stimulus_coord].values == target[self._stimulus_coord].values).all()
        assert np.array(prediction[self._neuroid_coord].values == target[self._neuroid_coord].values).all()
        # compute
        correlations = []
        for i in target[self._neuroid_coord].values:
            target_activations = target.sel(**{self._neuroid_coord: i}).squeeze()
            prediction_activations = prediction.sel(**{self._neuroid_coord: i}).squeeze()
            r, p = self._correlation(target_activations, prediction_activations)
            correlations.append(r)
        # package
        neuroid_dim = target[self._neuroid_coord].dims
        result = DataAssembly(correlations,
                              coords={coord: (dims, values)
                                      for coord, dims, values in walk_coords(target) if dims == neuroid_dim},
                              dims=neuroid_dim)
        return result
