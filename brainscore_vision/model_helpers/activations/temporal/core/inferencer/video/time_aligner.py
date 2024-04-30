import numpy as np
from brainio.assemblies import DataAssembly
from brainscore_vision.model_helpers.activations.temporal.inputs.video import Video


"""This module includes different time alignment strategies for the activations of a temporal neural network.

    A time alignment strategy is a function that takes the DataAssembly (with channel_temporal) and the video stimuli,
    and aligns the activations to the video time. The channel_temporal dimension will be changed into time_bin dimension,
    and the time_bin_start and time_bin_end will be added as coordinates of it.
"""


def _convert(assembly, time_bin_starts, time_bin_ends):
    # this function converts the "channel_temporal" dimension to "time_bin" dimension
    # if "channel_temporal" is not present, it adds a "time_bin" dimension
    asm_type = assembly.__class__
    if "channel_temporal" in assembly.dims:
        assembly = assembly.drop_vars("channel_temporal")
        assembly = assembly.rename({"channel_temporal": "time_bin"})
    else:
        assembly = assembly.expand_dims("time_bin")
    assembly = assembly.assign_coords({
        "time_bin_start": ("time_bin", time_bin_starts),
        "time_bin_end": ("time_bin", time_bin_ends)
    })
    return asm_type(assembly)

def estimate_layer_fps(assembly : DataAssembly, video : Video) -> DataAssembly:
    # in a temporal neural net, different layers may have different temporal resolutions
    # this function estimates the temporal resolution of a layer, based on the video fps
    fps = video.fps
    num_t = assembly.sizes['channel_temporal'] if "channel_temporal" in assembly.dims else 1
    duration = video.duration
    model_frame_interval = 1000 / fps
    estimated_frame_interval = duration / num_t
    estimated_multiplier = estimated_frame_interval / model_frame_interval
    estimated_multiplier = int(round(estimated_multiplier))  # round to nearest integer
    estimated_interval = model_frame_interval * estimated_multiplier
    time_bin_starts = np.arange(0, num_t) * estimated_interval
    time_bin_ends = time_bin_starts + estimated_interval
    return _convert(assembly, time_bin_starts, time_bin_ends)

def evenly_spaced(assembly : DataAssembly, video : Video) -> DataAssembly:
    # this function assumes that the activation of different time steps is evenly spaced
    num_t = assembly.sizes['channel_temporal'] if "channel_temporal" in assembly.dims else 1
    interval = video.duration / num_t
    time_bin_starts = np.linspace(0, video.duration, num_t+1)[:-1]
    time_bin_ends = time_bin_starts + interval
    time_bin_ends[-1] = video.duration
    return _convert(assembly, time_bin_starts, time_bin_ends)

def per_frame_aligned(assembly : DataAssembly, video : Video) -> DataAssembly:
    # this function assumes that the activation of different time steps is aligned with the video frames
    num_t = assembly.sizes['channel_temporal'] if "channel_temporal" in assembly.dims else 1
    assert video.num_frames <= num_t
    interval = 1000 / video.fps
    time_bin_starts = np.arange(0, num_t) * interval
    time_bin_ends = time_bin_starts + interval
    return _convert(assembly, time_bin_starts, time_bin_ends)

def ignore_time(assembly : DataAssembly, video : Video) -> DataAssembly:
    # this function treats the activations from the entire video as from a single time bin,
    # and treat the "channel_temporal" as a regular channel dimension and does no conversion 
    asm_type = assembly.__class__
    assembly = assembly.expand_dims("time_bin")
    time_bin_starts = [0]
    time_bin_ends = [video.duration]
    assembly = assembly.assign_coords({
        "time_bin_start": ("time_bin", time_bin_starts),
        "time_bin_end": ("time_bin", time_bin_ends)
    })
    return asm_type(assembly)