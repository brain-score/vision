from brainscore.benchmarks.public_benchmarks import FreemanZiembaV1PublicBenchmark, FreemanZiembaV2PublicBenchmark, \
    MajajV4PublicBenchmark, MajajITPublicBenchmark
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad
from model_tools.brain_transformation.temporal import TemporalIgnore
from .behavior import BehaviorArbiter, LogitsBehavior, ProbabilitiesMapping
from .neural import LayerMappedModel, LayerSelection, LayerScores


class ModelCommitment(BrainModel):
    """
    Standard process to convert a BaseModel (e.g. standard Machine Learning/Computer Vision model)
    into a BrainModel (e.g. commitments on layer-to-region, pixel-to-degrees, ...).
    """

    standard_region_benchmarks = {
        'V1': LazyLoad(FreemanZiembaV1PublicBenchmark),
        'V2': LazyLoad(FreemanZiembaV2PublicBenchmark),
        'V4': LazyLoad(MajajV4PublicBenchmark),
        'IT': LazyLoad(MajajITPublicBenchmark),
    }

    def __init__(self, identifier,
                 activations_model, layers, behavioral_readout_layer=None, region_benchmarks=None,
                 visual_degrees=8):
        self.layers = layers
        self.activations_model = activations_model
        self.region_benchmarks = {**self.standard_region_benchmarks, **(region_benchmarks or {})}
        layer_model = LayerMappedModel(identifier=identifier, activations_model=activations_model)
        self.layer_model = TemporalIgnore(layer_model)
        logits_behavior = LogitsBehavior(identifier=identifier, activations_model=activations_model)
        behavioral_readout_layer = behavioral_readout_layer or layers[-1]
        probabilities_behavior = ProbabilitiesMapping(identifier=identifier, activations_model=activations_model,
                                                      layer=behavioral_readout_layer)
        self.behavior_model = BehaviorArbiter({BrainModel.Task.label: logits_behavior,
                                               BrainModel.Task.probabilities: probabilities_behavior})
        self.do_behavior = False

        self._visual_degrees = visual_degrees

    def visual_degrees(self) -> int:
        return self._visual_degrees

    def start_task(self, task: BrainModel.Task, *args, **kwargs):
        if task != BrainModel.Task.passive:
            self.behavior_model.start_task(task, *args, **kwargs)
            self.do_behavior = True
        else:
            self.do_behavior = False

    def look_at(self, stimuli):
        if self.do_behavior:
            return self.behavior_model.look_at(stimuli)
        else:
            return self.layer_model.look_at(stimuli)

    def commit_region(self, region):
        layer_selection = LayerSelection(model_identifier=self.layer_model.identifier,
                                         activations_model=self.layer_model.activations_model, layers=self.layers,
                                         visual_degrees=self.visual_degrees())
        benchmark = self.region_benchmarks[region]
        best_layer = layer_selection(selection_identifier=region, benchmark=benchmark)
        self.layer_model.commit(region, best_layer)

    def start_recording(self, recording_target, time_bins):
        if recording_target not in self.layer_model.region_layer_map:  # not yet committed
            self.commit_region(recording_target)
        return self.layer_model.start_recording(recording_target, time_bins)

    @property
    def identifier(self):
        return self.layer_model.identifier
