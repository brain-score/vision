import logging

import numpy as np

from brainscore.metrics.regression import pls_regression
from brainscore.metrics.transformations import CartesianProduct
from brainscore.utils import fullname
from result_caching import store_xarray


class LayerSelection:
    def __init__(self, model_identifier, activations_model, layers):
        """
        :param model_identifier: this is separate from the container name because it only refers to
            the combination of (model, preprocessing), i.e. no mapping.
        """
        self.model_identifier = model_identifier
        self._activations_model = activations_model
        self.layers = layers
        self._logger = logging.getLogger(fullname(self))

    def map(self, training_benchmark, validation_benchmark):
        regression_ctr = pls_regression
        self._logger.debug("Finding best layer")
        best_layer = self._find_best_layer(
            regression_ctr=regression_ctr,
            training_benchmark=training_benchmark, validation_benchmark=validation_benchmark)

        self._logger.debug(f"Preparing mapped model using layer {best_layer}")
        regression = regression_ctr()
        mapping_activations = self._activations_model(layers=[best_layer],
                                                      stimuli=training_benchmark.assembly.stimulus_set)
        regression.fit(mapping_activations, training_benchmark.assembly)

        def mapped_model(test_stimuli):
            model_test_assembly = self._activations_model(layers=[best_layer], stimuli=test_stimuli)
            prediction = regression.predict(model_test_assembly)
            return prediction

        return mapped_model

    def _find_best_layer(self, regression_ctr,
                         training_benchmark, validation_benchmark):
        layer_scores = self._layer_scores(
            model_identifier=self.model_identifier, layers=self.layers, regression_ctr=regression_ctr,
            training_benchmark_name=training_benchmark.name, validation_benchmark_name=validation_benchmark.name,
            training_benchmark=training_benchmark, validation_benchmark=validation_benchmark)
        self._logger.debug(f"Layer scores (unceiled): {layer_scores.raw}")
        best_layer = layer_scores['layer'].values[layer_scores.sel(aggregation='center').argmax()]
        return best_layer

    @store_xarray(identifier_ignore=[
        'activations_func', 'regression_ctr', 'training_benchmark', 'validation_benchmark', 'layers'],
        combine_fields={'layers': 'layer'})
    def _layer_scores(self,
                      model_identifier, layers, training_benchmark_name, validation_benchmark_name,  # storage fields
                      regression_ctr, training_benchmark, validation_benchmark):
        training_activations = self._activations_model(layers=self.layers,
                                                       stimuli=training_benchmark.assembly.stimulus_set)
        # also run validation_activations to avoid running every layer separately
        self._activations_model(layers=self.layers, stimuli=validation_benchmark.assembly.stimulus_set)
        cross_layer = CartesianProduct(dividers=['layer'])

        def map_score(layer_training_activations):
            layer_regression = regression_ctr()
            layer_regression.fit(layer_training_activations, training_benchmark.assembly)
            layer = single_element(np.unique(layer_training_activations['layer']))

            def layer_model(stimulus_set):
                activations = self._activations_model(layers=[layer], stimuli=stimulus_set)
                prediction = layer_regression.predict(activations)
                return prediction

            score = validation_benchmark(layer_model)
            return score

        layer_scores = cross_layer(training_activations, apply=map_score)
        return layer_scores


def single_element(element_list):
    assert len(element_list) == 1
    return element_list[0]
