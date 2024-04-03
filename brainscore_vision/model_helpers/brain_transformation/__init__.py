from brainscore_vision import load_benchmark
from brainscore_vision.model_helpers.brain_transformation.temporal import TemporalIgnore
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import LazyLoad
from .behavior import BehaviorArbiter, LabelBehavior, ProbabilitiesMapping, OddOneOut
from .neural import LayerMappedModel, LayerSelection, LayerScores

STANDARD_REGION_BENCHMARKS = {
    'V1': LazyLoad(lambda: load_benchmark('FreemanZiemba2013public.V1-pls')),
    'V2': LazyLoad(lambda: load_benchmark('FreemanZiemba2013public.V2-pls')),
    'V4': LazyLoad(lambda: load_benchmark('MajajHong2015public.V4-pls')),
    'IT': LazyLoad(lambda: load_benchmark('MajajHong2015public.IT-pls')),
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
        # We set the visual degrees of the ActivationsExtractorHelper here to avoid changing its signature.
        #  The ideal solution would be to not expose the _extractor of the activations_model here, but to change
        #  the signature of the ActivationsExtractorHelper. See https://github.com/brain-score/vision/issues/554
        self.activations_model._extractor.set_visual_degrees(visual_degrees)  # for microsaccades
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
        odd_one_out = OddOneOut(identifier=identifier, activations_model=activations_model,
                                layer=behavioral_readout_layer)
        self.behavior_model = BehaviorArbiter({BrainModel.Task.label: logits_behavior,
                                               BrainModel.Task.probabilities: probabilities_behavior,
                                               BrainModel.Task.odd_one_out: odd_one_out,
                                               })
        self.do_behavior = False

    def visual_degrees(self) -> int:
        return self._visual_degrees

    def start_task(self, task: BrainModel.Task, *args, **kwargs):
        if task != BrainModel.Task.passive:
            self.behavior_model.start_task(task, *args, **kwargs)
            self.do_behavior = True
        else:
            self.do_behavior = False

    def look_at(self, stimuli, number_of_trials=1, require_variance: bool = False):
        if self.do_behavior:
            return self.behavior_model.look_at(stimuli, number_of_trials=number_of_trials,
                                               require_variance=require_variance)
        else:
            return self.layer_model.look_at(stimuli, number_of_trials=number_of_trials,
                                            require_variance=require_variance)

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
