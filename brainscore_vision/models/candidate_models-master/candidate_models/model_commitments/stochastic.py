from model_tools.brain_transformation import ModelCommitment, LayerScores, LayerMappedModel, LayerSelection, \
    RegionLayerMap, STANDARD_REGION_BENCHMARKS
import warnings
from brainscore.utils import LazyLoad
from brainscore.submission.utils import UniqueKeyDict

STOCHASTIC_MODELS = {'voneresnet-50': True, 'voneresnet-50-robust':True, 'voneresnet-50-non_stochastic':False}


class StochasticModelCommitment(ModelCommitment):
    """
    Similar to ModelCommitment but gets model activations multiple times depending on the number of trials. To be
    used with models that have stochastic activations.
    """

    def __init__(self, identifier, activations_model, layers, visual_degrees=8):
            layer_selection = StochasticLayerSelection(model_identifier=identifier,
                                                       activations_model=activations_model, layers=layers,
                                                       visual_degrees=visual_degrees)
            region_layer_map = RegionLayerMap(layer_selection=layer_selection,
                                              region_benchmarks=STANDARD_REGION_BENCHMARKS)
            super(StochasticModelCommitment, self).__init__(identifier=identifier, activations_model=activations_model,
                                                            layers=layers, region_layer_map=region_layer_map,
                                                            visual_degrees=visual_degrees)

    def look_at(self, stimuli, number_of_trials=1):
        stimuli_identifier = stimuli.identifier
        for trial_number in range(number_of_trials):
            if stimuli_identifier:
                stimuli.identifier = stimuli_identifier + '-trial' + f'{trial_number:03d}'
            if trial_number == 0:
                activations = super().look_at(stimuli, number_of_trials=1)
                if not activations.values.flags['WRITEABLE']:
                    activations.values.setflags(write=1)
            else:
                activations += super().look_at(stimuli, number_of_trials=1)
        stimuli.identifier = stimuli_identifier
        return activations/number_of_trials


def get_vonenet_commitment(identifier, activations_model, layers, visual_degrees=8, stochastic=True):
    if stochastic:
        model_commitment = StochasticModelCommitment(identifier=identifier, activations_model=activations_model,
                                                     layers=layers, visual_degrees=visual_degrees)
    else:
        model_commitment = ModelCommitment(identifier=identifier, activations_model=activations_model,
                                                     layers=layers, visual_degrees=visual_degrees)
    model_commitment.layer_model.region_layer_map['V1'] = 'vone_block.output'
    return model_commitment


class StochasticLayerSelection(LayerSelection):
    def __init__(self, model_identifier, activations_model, layers,  visual_degrees):
        super(StochasticLayerSelection, self).__init__(model_identifier=model_identifier,
                                                       activations_model=activations_model, layers=layers,
                                                       visual_degrees=visual_degrees)
        self._layer_scoring = StochasticLayerScores(model_identifier=model_identifier,
                                                    activations_model=activations_model,
                                                    visual_degrees=visual_degrees)


class StochasticLayerScores(LayerScores):
    def _create_mapped_model(self, region, layer, model, model_identifier, visual_degrees):
        return StochasticLayerMappedModel(identifier=f"{model_identifier}-{layer}", visual_degrees=visual_degrees,
                                activations_model=model, region_layer_map={region: layer})


class StochasticLayerMappedModel(LayerMappedModel):
    def run_activations(self, stimuli, layers, number_of_trials):
        stimuli_identifier = stimuli.identifier
        for trial_number in range(number_of_trials):
            if stimuli_identifier:
                stimuli.identifier = stimuli_identifier + '-trial' + f'{trial_number:03d}'
            if trial_number == 0:
                activations = self.activations_model(stimuli, layers=layers)
            else:
                activations += self.activations_model(stimuli, layers=layers)
        stimuli.identifier = stimuli_identifier
        return activations / number_of_trials


class VOneNetBrainPool(UniqueKeyDict):
    def __init__(self, base_model_pool, model_layers, reload=True):
        super(VOneNetBrainPool, self).__init__(reload)
        self.reload = True
        for basemodel_identifier, activations_model in base_model_pool.items():
            if basemodel_identifier not in model_layers:
                warnings.warn(f"{basemodel_identifier} not found in model_layers")
                continue
            model_layer = model_layers[basemodel_identifier]

            def load(identifier=basemodel_identifier, activations_model=activations_model, layers=model_layer):
                assert hasattr(activations_model, 'reload')
                activations_model.reload()
                brain_model = get_vonenet_commitment(identifier=identifier, activations_model=activations_model,
                                                            layers=layers, stochastic=STOCHASTIC_MODELS[identifier])
                return brain_model

            self[basemodel_identifier] = LazyLoad(load)
