import os
from brainio.stimuli import StimulusSet
import brainscore_vision
from brainscore_vision import model_registry, load_model

HOME_DIR = brainscore_vision.__path__[0]
TEST_DIR = os.path.join(HOME_DIR, "../tests/test_model_helpers/temporal")


def _build_stimulus_set():
    video_names=["dots1.mp4", "dots2.mp4"]
    stimulus_set = StimulusSet([{'stimulus_id': video_name, 'some_meta': video_name[::-1]}
                                for video_name in video_names])
    stimulus_set.stimulus_paths = {video_name: os.path.join(TEST_DIR, video_name)
                                   for video_name in video_names}
    return stimulus_set

load_model("r3d_18")

stimulus_set = _build_stimulus_set()
model = model_registry["r3d_18"]
# model.start_recording('IT', time_bins=[(100, 150), (150, 200), (200, 250)])
# model_response = model.look_at(stimulus_set)

base_model = model.activations_model
model_assembly = base_model(stimulus_set)