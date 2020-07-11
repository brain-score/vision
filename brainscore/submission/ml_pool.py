import warnings

from brainscore.utils import LazyLoad
from brainscore.submission.utils import UniqueKeyDict


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

regions = ['V1', 'V2', 'V4', 'IT']


class MLBrainPool(UniqueKeyDict):
    def __init__(self, base_model_pool, model_layers, vs_model_param=None):
        super(MLBrainPool, self).__init__()
        self.reload = True

        if vs_model_param is not None:
            target_model_pool = vs_model_param['tar_pool']
            stimuli_model_pool = vs_model_param['stim_pool']
            visual_search_layer = vs_model_param['model_layers']
            target_img_size = vs_model_param['tar_size']
            stimuli_img_size = vs_model_param['stim_size']
        else:
            target_model_pool = base_model_pool
            stimuli_model_pool = base_model_pool
            visual_search_layer = None
            target_img_size = None
            stimuli_img_size = None

        for (basemodel_identifier, activations_model), (target_model_identifier, target_model), (stimuli_model_identifier, stimuli_model) in zip(base_model_pool.items(), target_model_pool.items(), stimuli_model_pool.items()):
            if basemodel_identifier not in model_layers:
                warnings.warn(f"{basemodel_identifier} not found in model_layers")
                continue
            layers = model_layers[basemodel_identifier]

            from model_tools.brain_transformation import ModelCommitment
            # enforce early parameter binding: https://stackoverflow.com/a/3431699/2225200

            def load(identifier=basemodel_identifier, activations_model=activations_model, layers=layers):
                assert hasattr(activations_model, 'reload')
                activations_model.reload()

                search_target_model_param = {}
                search_stimuli_model_param = {}
                if (vs_model_param is not None) and (basemodel_identifier == 'vgg-16'): #as vs_layer is implemented only for vgg-16 as of now
                    search_target_model_param['target_model'] = target_model
                    search_stimuli_model_param['stimuli_model'] = stimuli_model
                    search_target_model_param['target_layer'] = visual_search_layer[basemodel_identifier][0]
                    search_stimuli_model_param['stimuli_layer'] = visual_search_layer[basemodel_identifier][0]
                    search_target_model_param['target_img_size'] = target_img_size
                    search_stimuli_model_param['search_image_size'] = search_image_size
                else:
                    search_target_model_param['target_model'] = None
                    search_stimuli_model_param['stimuli_model'] =  None
                    search_target_model_param['target_layer'] = None
                    search_stimuli_model_param['stimuli_layer'] = None
                    search_target_model_param['target_img_size'] = None
                    search_stimuli_model_param['search_image_size'] = None

                brain_model = ModelCommitment(identifier=identifier, activations_model=activations_model,
                                              layers=layers,
                                              search_target_model_param=search_target_model_param,
                                              search_stimuli_model_param=search_stimuli_model_param)
                for region in regions:
                    brain_model.commit_region(region)
                return brain_model

            self[basemodel_identifier] = LazyLoad(load)
