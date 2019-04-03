from brainscore.model_interface import BrainModel
from .neural import LayerModel, LayerSelection, LayerScores
from .stimuli import PixelsToDegrees


class ModelCommitment(BrainModel):
    def __init__(self, identifier, activations_model, layers):
        self.layers = layers
        self.region_assemblies = {}
        self.layer_model = LayerModel(identifier=identifier, activations_model=activations_model)
        # forward brain-interface methods
        self.look_at = self.layer_model.look_at
        self.start_task = self.layer_model.start_task

    def commit_region(self, region, assembly, assembly_stratification=None):
        self.region_assemblies[region] = (assembly, assembly_stratification)  # lazy, only run when actually needed

    def do_commit_region(self, region):
        layer_selection = LayerSelection(model_identifier=self.layer_model.identifier,
                                         activations_model=self.layer_model.activations_model, layers=self.layers)
        assembly, assembly_stratification = self.region_assemblies[region]
        best_layer = layer_selection(assembly, assembly_stratification=assembly_stratification)
        self.layer_model.commit(region, best_layer)

    def start_recording(self, recording_target):
        if recording_target not in self.layer_model.region_layer_map:  # not yet committed
            self.do_commit_region(recording_target)
        return self.layer_model.start_recording(recording_target)

    @property
    def identifier(self):
        return self.layer_model.identifier
