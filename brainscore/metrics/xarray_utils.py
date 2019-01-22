import numpy as np

from brainio_base.assemblies import DataAssembly
from brainscore.assemblies import walk_coords


class Defaults:
    stimulus_coord = 'image_id'
    neuroid_dim = 'neuroid'
    neuroid_coord = 'neuroid_id'


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
        # compute correlation per neuroid
        correlations = []
        for i in target[self._neuroid_coord].values:
            target_neuroids = target.sel(**{self._neuroid_coord: i}).squeeze()
            prediction_neuroids = prediction.sel(**{self._neuroid_coord: i}).squeeze()
            r, p = self._correlation(target_neuroids, prediction_neuroids)
            correlations.append(r)
        # package
        neuroid_dim = target[self._neuroid_coord].dims
        result = DataAssembly(correlations,
                              coords={coord: (dims, values)
                                      for coord, dims, values in walk_coords(target) if dims == neuroid_dim},
                              dims=neuroid_dim)
        return result
        # neuroid_coords = {coord: (dims, values) for coord, dims, values in walk_coords(target) if dims == neuroid_dim}
        # dim = neuroid_dim
        # if len(neuroid_coords) == 1:
        #     dim = next(iter(neuroid_coords))
        #     neuroid_coords = {coord: (dim, values) for coord, (dims, values) in neuroid_coords.items()}
        # result = DataAssembly(correlations, coords=neuroid_coords, dims=dim)
        # if len(neuroid_coords) == 1:
        #     result = result.stack(**{neuroid_dim[0]: [dim]})
