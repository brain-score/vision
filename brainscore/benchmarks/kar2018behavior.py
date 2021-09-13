import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics.image_level_behavior import I2n
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad


class DicarloKar2018cocoI2n(BenchmarkBase):
    def __init__(self):
        self._metric = I2n(sample_object_coord='sample_object', distractor_object_coord='distractor_object')
        self._fitting_stimuli = LazyLoad(lambda: brainscore.get_stimulus_set('dicarlo.Kar2018coco_color.public'))
        self._assembly = LazyLoad(lambda: brainscore.get_assembly('dicarlo.Kar2018coco_behavior.private'))
        self._visual_degrees = 8
        self._number_of_trials = 2
        super(DicarloKar2018cocoI2n, self).__init__(
            identifier='dicarlo.Kar2018coco-i2n',
            version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly, skipna=True),
            parent='behavior',
            bibtex=None,  # unpublished
        )

    def __call__(self, candidate: BrainModel):
        fitting_stimuli = place_on_screen(self._fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        probabilities = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        score = self._metric(probabilities, self._assembly)
        score = self._metric.ceil_score(score, self.ceiling)
        return score
