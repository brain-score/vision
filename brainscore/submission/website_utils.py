"""
Helper functions to provide meta information to the website.
"""
import fire
import logging
import numpy as np
import sys
from PIL import Image
from numpy.random import RandomState
from pathlib import Path
from tqdm import tqdm
from typing import Union, List, Tuple

from brainio.assemblies import NeuroidAssembly
from brainio.stimuli import StimulusSet
from brainscore import benchmark_pool
from brainscore.model_interface import BrainModel

_logger = logging.getLogger(__name__)


class ImageStorerDummyModel(BrainModel):
    def __init__(self):
        self._time_bins = None
        self.stimuli = None

    @property
    def identifier(self) -> str:
        return 'imagestorer-dummymodel'

    def visual_degrees(self) -> int:
        return 8

    def look_at(self, stimuli: Union[StimulusSet, List[str]], number_of_trials=1):
        if len(stimuli) == 1:  # configuration stimuli, e.g. Kar2019 or Marques2020. Return to get to the real stimuli
            return NeuroidAssembly([[np.arange(len(self._time_bins))]], coords={
                **{'neuroid_id': ('neuroid', [123]), 'neuroid_num': ('neuroid', [123])},
                **{column: ('presentation', values) for column, values in stimuli.iteritems()},
                **{'time_bin_start': ('time_bin', [start for start, end in self._time_bins]),
                   'time_bin_end': ('time_bin', [end for start, end in self._time_bins])},
            }, dims=['presentation', 'neuroid', 'time_bin'])
        self.stimuli = stimuli
        raise StopIteration()

    def start_task(self, task: BrainModel.Task, fitting_stimuli=None):
        pass

    def start_recording(self, recording_target: BrainModel.RecordingTarget, time_bins=List[Tuple[int]]):
        self._time_bins = time_bins


def sample_benchmark_images(num_samples: int = 30, replace=False):
    image_directory = Path(__file__).parent / 'sample_images'

    image_storer = ImageStorerDummyModel()
    for benchmark_identifier, benchmark in tqdm(benchmark_pool.items(), desc='benchmarks'):
        benchmark_specifier = f"{benchmark_identifier}_v{benchmark.version}"
        _logger.debug(f"Benchmark {benchmark_specifier}")
        benchmark_directory = image_directory / benchmark_specifier
        if benchmark_directory.is_dir() and not replace:
            _logger.debug(f"Skipping {benchmark_directory} since it already exists and replace is {replace}")
            continue
        benchmark_directory.mkdir(exist_ok=True, parents=True)
        try:
            benchmark(image_storer)
        except StopIteration:
            pass
        stimuli = image_storer.stimuli
        random_state = RandomState(len(stimuli))  # seed with number of stimuli to pick different indices per benchmark
        sample_image_ids = random_state.choice(stimuli['image_id'],
                                               size=num_samples,
                                               replace=False if len(stimuli) >= num_samples else True)
        for sample_number, image_id in enumerate(sample_image_ids):
            source_path = Path(stimuli.get_image(image_id))
            image = Image.open(source_path)
            target_path = benchmark_directory / f"{sample_number}.png"
            image.save(target_path)  # convert to png


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    for shush_logger in ['PIL']:
        logging.getLogger(shush_logger).setLevel(logging.INFO)
    fire.Fire()
