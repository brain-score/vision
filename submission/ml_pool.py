import warnings

import itertools

from model_tools.brain_transformation import ModelCommitment

from brainscore.utils import LazyLoad
from submission.utils import UniqueKeyDict


class ModelLayers(UniqueKeyDict):
    def __init__(self, layers):
        super(ModelLayers, self).__init__()
        for basemodel_identifier, default_layers in layers.items():
            self[basemodel_identifier] = default_layers

    @staticmethod
    def _item(item):
        return item

    def __getitem__(self, item):
        return super(ModelLayers, self).__getitem__(self._item(item))

    def __contains__(self, item):
        return super(ModelLayers, self).__contains__(self._item(item))


class ModelLayersPool(UniqueKeyDict):
    def __init__(self, base_model_pool, model_layers):
        super(ModelLayersPool, self).__init__()
        for basemodel_identifier, activations_model in base_model_pool.items():
            if basemodel_identifier not in model_layers:
                warnings.warn(f"{basemodel_identifier} not found in model_layers")
                continue
            layers = model_layers[basemodel_identifier]

            # for identifier, activations_model in Hooks().iterate_hooks(basemodel_identifier, activations_model):
            self[basemodel_identifier] = {'model': activations_model, 'layers': layers}


regions = ['V1', 'V2', 'V4', 'IT']


class MLBrainPool(UniqueKeyDict):
    def __init__(self, base_model_pool, model_layers):
        super(MLBrainPool, self).__init__()

        for basemodel_identifier, activations_model in base_model_pool.items():
            if basemodel_identifier not in model_layers:
                warnings.warn(f"{basemodel_identifier} not found in model_layers")
                continue
            layers = model_layers[basemodel_identifier]

            self[basemodel_identifier] = LazyLoad(
                lambda identifier=basemodel_identifier, activations_model=activations_model, layers=layers:
                ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers))
