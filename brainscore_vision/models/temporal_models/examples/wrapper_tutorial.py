# This is a tutorial for how to use model helpers for the temporal models.
# There will be multiple "breakpoint" in this script, where the code will stop and wait for user input.
# Type "c" to continue to the next breakpoint, and "q" to quit the tutorial.
# Type "l" to show the current line.
# You can also type "n" to skip the current breakpoint and go to the next one, or type "p {variable}"
# to print the value of a variable.



"""Step 1: Import necessary modules"""

# All temporal model helpers are in brainscore_vision.model_helpers.activations.temporal
import brainscore_vision.model_helpers.activations.temporal

# In temporal.model, the wrapper class that converts a ML model to a basic temporal model is defined.
# Currently, the only supported library is PyTorch, and the corresponding class is PytorchWrapper
from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper
import torch
from torch import nn

breakpoint()
# We next show how to use this class to wrap a PyTorch model



"""Step 2: Wrap a model"""

# To wrap a pytorch model, it is necessary to at least several things:

# - the transform_input function is used to convert a Stimulus (here a Video) to the input format that the model expects
from brainscore_vision.model_helpers.activations.temporal.inputs import Video
def transform_input(input: Video):
    arr = input.to_numpy()  # [T, H, W, C]
    arr = arr / 255.0  # normalize
    arr = torch.Tensor(arr)
    arr = arr.permute(3, 0, 1, 2)  # [C, T, H, W]
    return arr

# - the model is the actual ML model that takes a list of inputs in the expected format and does batch forward pass
torch.manual_seed(42)
class DummyVideoModel(nn.Module):
    def __init__(self):
        super(DummyVideoModel, self).__init__()
        self.layer1 = nn.Conv3d(3, 3, kernel_size=3, stride=5, padding=1)
        self.layer2 = nn.Conv3d(3, 3, kernel_size=3, stride=5, padding=1)
        self.layer3 = nn.Conv3d(3, 3, kernel_size=3, stride=5, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = DummyVideoModel()

# - the activations_spec is a dictionary that specifies the format of the activations for each layer
activations_spec = {
    "layer1": "CTHW",
    "layer2": "CTHW",
    "layer3": "CTHW",
}

# - fps is the number of frames per second that the model expects
fps = 25

# Finally, the PytorchWrapper is initialized with the above parameters
base_model = PytorchWrapper(identifier="model_name", model=model, preprocessing=transform_input, fps=fps, activations_spec=activations_spec)
breakpoint()



"""Step 3: Extract activations"""

import os
import brainscore_vision
HOME_DIR = brainscore_vision.__path__[0]
TEST_DIR = os.path.join(HOME_DIR, "../tests/test_model_helpers/temporal")

# The base_model object expects a list of Video objects as input, and a list of layer names for which activations will be extracted
video_paths = [os.path.join(TEST_DIR, "dots1.mp4"), os.path.join(TEST_DIR, "dots2.mp4")]  
layers = ["layer1", "layer2", "layer3"] 

# It returns a NeuroidAssembly object, which is an xarray DataArray with additional metadata
# The dimensions are [stimulus_path, time_bin, neuroid]
model_assembly = base_model(video_paths, layers)
print(model_assembly)

# <xarray.NeuroidAssembly (stimulus_path: 2, neuroid: 180012, time_bin: 150)>
# array([[[-0.06396484, -0.06396484, -0.06396484, ...,         nan,
#                  nan,         nan],
#         [-0.06396484, -0.06396484, -0.06396484, ...,         nan,
#                  nan,         nan],
#         [-0.06396484, -0.06396484, -0.06396484, ...,         nan,
#                  nan,         nan],
#         ...,
#         [        nan,         nan,         nan, ...,         nan,
#                  nan,         nan],
#         [        nan,         nan,         nan, ...,         nan,
#                  nan,         nan],
#         [        nan,         nan,         nan, ...,         nan,
#                  nan,         nan]],
#         ...,
#         [-0.03677368, -0.03677368, -0.03677368, ..., -0.04251099,
#          -0.04251099, -0.04251099],
#         [-0.03677368, -0.03677368, -0.03677368, ..., -0.04251099,
#          -0.04251099, -0.04251099],
#         [-0.03677368, -0.03677368, -0.03677368, ..., -0.04251099,
#          -0.04251099, -0.04251099]]])
# Coordinates:
#   * stimulus_path   (stimulus_path) <U110 '/home/ytang/workspace/modules/tmp/...
#   * neuroid         (neuroid) MultiIndex
#   - neuroid_id      (neuroid) object 'layer1.0' 'layer1.1' ... 'layer3.299'
#   - neuroid_num     (neuroid) int64 0 1 2 3 4 5 6 ... 294 295 296 297 298 299
#   - channel_x       (neuroid) int64 0 1 2 3 4 5 6 7 8 9 ... 0 1 2 3 4 5 6 7 8 9
#   - channel_y       (neuroid) int64 0 0 0 0 0 0 0 0 0 0 ... 9 9 9 9 9 9 9 9 9 9
#   - layer           (neuroid) object 'layer1' 'layer1' ... 'layer3' 'layer3'
#   - channel         (neuroid) int64 0 0 0 0 0 0 0 0 0 0 ... 2 2 2 2 2 2 2 2 2 2
#   * time_bin        (time_bin) MultiIndex
#   - time_bin_start  (time_bin) float64 0.0 40.0 80.0 ... 5.92e+03 5.96e+03
#   - time_bin_end    (time_bin) float64 40.0 80.0 120.0 ... 5.96e+03 6e+03
breakpoint()

# In the above assembly, the dimensions are [stimulus_path, neuroid, time_bin].
# - The 2 videos passed in have been concatenated along the stimulus_path dimension.
# - The activations for each layer and for each channel have been concatenated along the neuroid dimension.
#   These flatten channels collectively make up 180012 neuroids.
# - Finally, activations from all layers are temporally aligned to the fps of the model, which is 25.
#   You can see the time_bins are all of length 40.0ms, which is 1000ms / 25.

# NOTE: there are some NaNs in the assembly. This is because the two videos have different durations.
#       However, we choose to concatenate them along the time_bin dimension, so the shorter video will have NaNs at the end.



"""Step 4: Choose a different inferencer"""

# When a PytorchWrapper is initialized, it uses the default inferencer, which is TemporalInferencer.
# This inferencer maps the time dimension of the output activations to the actual time stamps from the video.
# Please see the documentation of TemporalInferencer for more details.

# However, if you want to use another inferencer, you can rebuild it into the wrapper.
from brainscore_vision.model_helpers.activations.temporal.core.video import CausalInferencer
base_model.build_extractor(CausalInferencer, fps=fps, activations_spec=activations_spec)

model_assembly = base_model(video_paths, layers)
# The CausalInferencer passes the video in a frame-by-frame manner, so it costs more time to extract activations.
# Please see the documentation of CausalInferencer for more details.

# It is also possible to write your own inferencer.
breakpoint()



"""Step 5: Bypassing the inferencer"""

# If you don't want to use the time alignment and inference properties of a inferencer, it is also possible 
# to bypass the above wrapping procedure and directly get the model activations as raw, un-flatten np arrays.

# You can use the BatchExecutor class to do this.
from brainscore_vision.model_helpers.activations.temporal.core.executor import BatchExecutor
executor = BatchExecutor(base_model.get_activations, base_model.preprocessing, 
                         batch_size=1, batch_grouper=lambda v: v.duration, batch_padding=False)

video_stimuli = [Video.from_path(video_path) for video_path in video_paths]
executor.add_stimuli(video_stimuli)
activations = executor.execute(layers)  # return {layer: np.array}
print(activations)  
breakpoint()

