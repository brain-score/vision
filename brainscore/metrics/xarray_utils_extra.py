import numpy as np

class Defaults:
    expected_dims = ('presentation', 'neuroid')
    stimulus_coord = 'image_id'
    neuroid_dim = 'neuroid'
    neuroid_coord = 'neuroid_id'


class XarrayPearson:
    def __init__(self, correlation_coord=Defaults.stimulus_coord, neuroid_coord=Defaults.neuroid_coord):
        self._correlation_coord = correlation_coord
        self._neuroid_coord = neuroid_coord

    def __call__(self, prediction, target):
        # align
        prediction = prediction.sortby([self._correlation_coord, self._neuroid_coord])
        target = target.sortby([self._correlation_coord, self._neuroid_coord])
        assert np.array(prediction[self._correlation_coord].values == target[self._correlation_coord].values).all()
        assert np.array(prediction[self._neuroid_coord].values == target[self._neuroid_coord].values).all()

        # Standardize
        correlation_dim = prediction.coords[self._correlation_coord].dims
        assert(len(correlation_dim))
        correlation_dim = correlation_dim[0]
        prediction = (prediction - prediction.mean(dim=correlation_dim)) / prediction.std(dim=correlation_dim)
        target = (target - target.mean(dim=correlation_dim)) / target.std(dim=correlation_dim)

        # Correlation
        product = np.multiply(prediction, target)
        summed = product.sum(dim=correlation_dim)
        pearson_corr = summed/prediction.sizes[correlation_dim]

        return pearson_corr


