
from candidate_models.model_commitments.cornets import CORnetCommitment, _build_time_mappings, CORNET_S_TIMEMAPPING
import logging

from candidate_models.base_models import vonecornet
from brainscore.submission.utils import UniqueKeyDict
from brainscore.utils import LazyLoad
_logger = logging.getLogger(__name__)


class VOneCORnetCommitment(CORnetCommitment):
    def start_recording(self, recording_target, time_bins):
        if recording_target == 'V1':
            self.recording_layers = ['vone_block.output-t0']
        else:
            self.recording_layers = [layer for layer in self.layers if recording_target in layer]
        self.recording_time_bins = time_bins


def vonecornet_s_brainmodel():
    # map region -> (time_start, time_step_size, timesteps)
    time_mappings = CORNET_S_TIMEMAPPING
    return VOneCORnetCommitment(identifier='CORnet-S', activations_model=vonecornet('cornets'),
                                layers=['vone_block.output-t0'] + [f'model.{area}.output-t{timestep}'
                                                                   for area, timesteps in
                                                                   [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                                                   for timestep in timesteps] +
                                       ['model.decoder.avgpool-t0'], time_mapping=_build_time_mappings(time_mappings))


class VOneCORnetBrainPool(UniqueKeyDict):
    def __init__(self):
        super(VOneCORnetBrainPool, self).__init__(reload=True)

        model_pool = {
            'VOneCORnet-S': LazyLoad(vonecornet_s_brainmodel),
        }

        self._accessed_brain_models = []

        for identifier, brain_model in model_pool.items():
            self[identifier] = brain_model


vonecornet_brain_pool = VOneCORnetBrainPool()
