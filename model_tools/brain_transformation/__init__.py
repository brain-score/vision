from brainscore.model_interface import BrainModel
from model_tools.brain_transformation.temporal import TemporalIgnore
from .behavior import LogitsBehavior
from .neural import LayerMappedModel, LayerSelection, LayerScores
from .stimuli import PixelsToDegrees


class ModelCommitment(BrainModel):
    def __init__(self, identifier, activations_model, layers):
        self.layers = layers
        self.region_assemblies = {}
        layer_model = LayerMappedModel(identifier=identifier, activations_model=activations_model)
        self.layer_model = TemporalIgnore(layer_model)
        self.behavior_model = LogitsBehavior(identifier=identifier, activations_model=activations_model)
        self.do_behavior = False

    def start_task(self, task: BrainModel.Task):
        if task != BrainModel.Task.passive:
            self.behavior_model.start_task(task)
            self.do_behavior = True

    def look_at(self, stimuli):
        if self.do_behavior:
            return self.behavior_model.look_at(stimuli)
        else:
            return self.layer_model.look_at(stimuli)

    def commit_region(self, region, assembly, assembly_stratification=None):
        self.region_assemblies[region] = (assembly, assembly_stratification)  # lazy, only run when actually needed

    def do_commit_region(self, region):
        layer_selection = LayerSelection(model_identifier=self.layer_model.identifier,
                                         activations_model=self.layer_model.activations_model, layers=self.layers)
        assembly, assembly_stratification = self.region_assemblies[region]
        best_layer = layer_selection(assembly, assembly_stratification=assembly_stratification)
        self.layer_model.commit(region, best_layer)

    def start_recording(self, recording_target, time_bins):
        if recording_target not in self.layer_model.region_layer_map:  # not yet committed
            self.do_commit_region(recording_target)
        return self.layer_model.start_recording(recording_target, time_bins)

    @property
    def identifier(self):
        return self.layer_model.identifier
