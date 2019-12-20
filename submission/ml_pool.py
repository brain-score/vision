import warnings

import itertools

from brainscore.utils import LazyLoad
from submission.utils import UniqueKeyDict


class Hooks:
    HOOK_SEPARATOR = "--"

    def __init__(self):
        pca_components = 1000
        from model_tools.activations.pca import LayerPCA
        from model_tools.brain_transformation import PixelsToDegrees
        self.activation_hooks = {
            f"pca_{pca_components}": lambda activations_model: LayerPCA.hook(
                activations_model, n_components=pca_components),
            "degrees": lambda activations_model: PixelsToDegrees.hook(
                activations_model, target_pixels=activations_model.image_size)}

    def iterate_hooks(self, basemodel_identifier, activations_model):
        for hook_identifiers in itertools.chain.from_iterable(
                itertools.combinations(self.activation_hooks, n) for n in range(len(self.activation_hooks) + 1)):
            hook_identifiers = list(sorted(hook_identifiers))
            identifier = basemodel_identifier
            if len(hook_identifiers) > 0:
                identifier += self.HOOK_SEPARATOR + "-".join(hook_identifiers)

            # enforce early parameter binding: https://stackoverflow.com/a/3431699/2225200
            def load(identifier=identifier, activations_model=activations_model,
                     hook_identifiers=hook_identifiers):
                activations_model.identifier = identifier  # since inputs are different, also change identifier
                for hook in hook_identifiers:
                    self.activation_hooks[hook](activations_model)
                return activations_model

            yield identifier, LazyLoad(load)


class ModelLayers(UniqueKeyDict):
    def __init__(self, layers):
        super(ModelLayers, self).__init__()
        for basemodel_identifier, default_layers in layers.items():
            self[basemodel_identifier] = default_layers

    @staticmethod
    def _item(item):
        # if item.startswith('mobilenet'):
        #     return "_".join(item.split("_")[:2])
        # if item.startswith('bagnet'):
        #     return 'bagnet'
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

            for identifier, activations_model in Hooks().iterate_hooks(basemodel_identifier, activations_model):
                self[identifier] = {'model': activations_model, 'layers': layers}


regions = ['V1', 'V2', 'V4', 'IT']


class MLBrainPool(UniqueKeyDict):
    def __init__(self, base_model_pool, model_layers):
        super(MLBrainPool, self).__init__()

        for basemodel_identifier, activations_model in base_model_pool.items():
            if basemodel_identifier not in model_layers:
                warnings.warn(f"{basemodel_identifier} not found in model_layers")
                continue
            layers = model_layers[basemodel_identifier]

            for identifier, activations_model in Hooks().iterate_hooks(basemodel_identifier, activations_model):
                if identifier in self:  # already pre-defined
                    continue
                from model_tools.brain_transformation import ModelCommitment
                # enforce early parameter binding: https://stackoverflow.com/a/3431699/2225200
                def load(identifier=identifier, activations_model=activations_model, layers=layers):
                    brain_model = ModelCommitment(identifier=identifier, activations_model=activations_model,
                                                  layers=layers)
                    for region in regions:
                        brain_model.commit_region(region)
                    return brain_model

                self[identifier] = LazyLoad(load)
