import numpy as np

def estimate_layer_fps(num_t, input):
    # in a temporal neural net, different layers may have different temporal resolutions
    # this function estimates the temporal resolution of a layer, based on the input fps
    fps = input.fps
    duration = input.duration
    model_frame_interval = 1000 / fps
    estimated_frame_interval = duration / num_t
    estimated_multiplier = estimated_frame_interval / model_frame_interval
    estimated_multiplier = int(round(estimated_multiplier))  # round to nearest integer
    estimated_interval = model_frame_interval * estimated_multiplier
    time_bin_starts = np.arange(0, num_t) * estimated_interval
    time_bin_ends = time_bin_starts + estimated_interval
    return time_bin_starts, time_bin_ends

def evenly_spaced(num_t, input):
    # this function assumes that the activation of different time steps is evenly spaced
    interval = input.duration / num_t
    time_bin_starts = np.arange(0, num_t) * interval
    time_bin_ends = time_bin_starts + interval
    return time_bin_starts, time_bin_ends

def per_frame_aligned(num_t, input):
    assert input.num_frames == num_t
    interval = 1000 / input.fps
    time_bin_starts = np.arange(0, num_t) * interval
    time_bin_ends = time_bin_starts + interval
    return time_bin_starts, time_bin_ends

def ignore_time(num_t, input):
    assert num_t == 1
    time_bins_starts = [input.start_time]
    time_bins_ends = [input.end_time]
    return time_bins_starts, time_bins_ends