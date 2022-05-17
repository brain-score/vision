from brainscore.benchmarks.public_benchmarks import FreemanZiembaV1PublicBenchmark, FreemanZiembaV2PublicBenchmark, \
    MajajHongV4PublicBenchmark, MajajHongITPublicBenchmark
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad
from model_tools.brain_transformation.temporal import TemporalIgnore
from .behavior import BehaviorArbiter, LabelBehavior, ProbabilitiesMapping
from .neural import LayerMappedModel, LayerSelection, LayerScores

STANDARD_REGION_BENCHMARKS = {
    'V1': LazyLoad(FreemanZiembaV1PublicBenchmark),
    'V2': LazyLoad(FreemanZiembaV2PublicBenchmark),
    'V4': LazyLoad(MajajHongV4PublicBenchmark),
    'IT': LazyLoad(MajajHongITPublicBenchmark),
}


class ModelCommitment(BrainModel):
    """
    Standard process to convert a BaseModel (e.g. standard Machine Learning/Computer Vision model)
    into a BrainModel (e.g. commitments on layer-to-region, pixel-to-degrees, ...).
    """

    def __init__(self, identifier,
                 activations_model, layers, behavioral_readout_layer=None, region_layer_map=None,
                 visual_degrees=8):
        self.layers = layers
        self.activations_model = activations_model
        self._visual_degrees = visual_degrees
        # region-layer mapping
        if region_layer_map is None:
            layer_selection = LayerSelection(model_identifier=identifier,
                                             activations_model=activations_model, layers=layers,
                                             visual_degrees=visual_degrees)
            region_layer_map = RegionLayerMap(layer_selection=layer_selection,
                                              region_benchmarks=STANDARD_REGION_BENCHMARKS)
        # neural
        layer_model = LayerMappedModel(identifier=identifier, activations_model=activations_model,
                                       region_layer_map=region_layer_map)
        self.layer_model = TemporalIgnore(layer_model)
        logits_behavior = LabelBehavior(identifier=identifier, activations_model=activations_model)
        behavioral_readout_layer = behavioral_readout_layer or layers[-1]
        probabilities_behavior = ProbabilitiesMapping(identifier=identifier, activations_model=activations_model,
                                                      layer=behavioral_readout_layer)
        self.behavior_model = BehaviorArbiter({BrainModel.Task.label: logits_behavior,
                                               BrainModel.Task.probabilities: probabilities_behavior})
        self.do_behavior = False

    def visual_degrees(self) -> int:
        return self._visual_degrees

    def start_task(self, task: BrainModel.Task, *args, **kwargs):
        if task != BrainModel.Task.passive:
            self.behavior_model.start_task(task, *args, **kwargs)
            self.do_behavior = True
        else:
            self.do_behavior = False

    def look_at(self, stimuli, number_of_trials=1):
        if self.do_behavior:
            return self.behavior_model.look_at(stimuli, number_of_trials=number_of_trials)
        else:
            return self.layer_model.look_at(stimuli, number_of_trials=number_of_trials)

    def start_recording(self, recording_target, time_bins):
        return self.layer_model.start_recording(recording_target, time_bins)

    @property
    def identifier(self):
        return self.layer_model.identifier


class RegionLayerMap(dict):
    """
    mapping of regions to layers that only evaluates lazily and only once.
    """

    def __init__(self, layer_selection, region_benchmarks):
        super(RegionLayerMap, self).__init__()
        self.layer_selection = layer_selection
        self.region_benchmarks = region_benchmarks

    def __getitem__(self, region):
        if region not in self:  # not yet committed
            self.commit_region(region)
        return super(RegionLayerMap, self).__getitem__(region)

    def commit_region(self, region):
        benchmark = self.region_benchmarks[region]
        best_layer = self.layer_selection(selection_identifier=region, benchmark=benchmark)
        self[region] = best_layer
