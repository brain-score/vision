import xarray as xr
import copy
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
from PIL import Image
from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet

from brainscore_vision import BrainModel


class PixelModel(BrainModel):
    def __init__(self):
        super(PixelModel, self).__init__()
        # where and when to record
        self.neural_recordings: List[Tuple[BrainModel.RecordingTarget, List[Tuple[int, int]]]] = []
        self.behavioral_task: Union[None, BrainModel.Task] = None
        self.behavioral_readout = None

    @property
    def identifier(self):
        return "pixels"

    def visual_degrees(self) -> int:
        return 8

    def look_at(self, stimuli: Union[StimulusSet, List[str]], number_of_trials=1):
        activations = self._pixels_from_stimuli(stimuli)

        output = {'behavior': [], 'neural': []}
        # behavior
        if self.behavioral_task:
            behavioral_output = self.behavioral_readout(activations)
            output['behavior'] = behavioral_output
        # neural
        for recording_target, time_bins in self.neural_recordings:
            region_activations = copy.deepcopy(activations)
            region_activations['region'] = 'neuroid', [recording_target] * len(region_activations['neuroid'])
            output['neural'].append(region_activations)
        output['neural'] = xr.concat(output['neural'], dim='presentation')
        return output

    def start_task(self, task: BrainModel.Task, fitting_stimuli: StimulusSet):
        activations = self._pixels_from_stimuli(fitting_stimuli)
        self.behavioral_readout = ...  # TODO: import linear readout from model_helpers, fit on pixel activations

    def start_recording(self, recording_target: BrainModel.RecordingTarget, time_bins: List[Tuple[int, int]]):
        self.neural_recordings.append((recording_target, time_bins))

    def _pixels_from_stimuli(self, stimuli: Union[StimulusSet, List[Union[Path, str]]]) -> NeuroidAssembly:
        paths = stimuli if not isinstance(stimuli, StimulusSet) else \
            [stimuli.get_stimulus(stimulus_id) for stimulus_id in stimuli['stimulus_id']]
        pixels = np.array([self._pixels_from_image(path) for path in paths])
        pixels_flat = np.reshape(pixels, [len(stimuli), -1])
        num_activations = pixels_flat.shape[-1]
        assembly = NeuroidAssembly(pixels_flat, coords={
            'stimulus_path': ('stimulus_path', paths),
            'neuroid_num': ('neuroid', np.arange(num_activations)),
            'model': ('neuroid', [self.identifier] * num_activations),
            'layer': ('neuroid', [self.identifier] * num_activations),
        }, dims=['stimulus_path', 'neuroid'])
        self._add_neuroid_id(assembly)
        if isinstance(stimuli, StimulusSet):
            assembly = attach_stimulus_set_meta(assembly, stimuli)  # TODO: import from model_helpers
        else:
            assembly = assembly.stack(presentation=['stimulus_path'])
        return assembly

    def _pixels_from_image(self, path: Union[str, Path]):
        image = Image.open(path)
        image = image.convert('RGB')  # make sure everything is in RGB and not grayscale L
        image = image.resize((256, 256))  # resize all images to same size
        return np.array(image)

    def _add_neuroid_id(self, assembly: NeuroidAssembly, from_coords=('model', 'layer', 'neuroid_num')):
        # TODO: move this method into model_helpers
        neuroid_id = [".".join([f"{value}" for value in values]) for values in zip(*[
            assembly[coord].values for coord in from_coords])]
        assembly['neuroid_id'] = 'neuroid', neuroid_id
