import numpy as np

from brainscore_vision.model_helpers.brain_transformation.temporal import assembly_time_align


def parallelize(func, iterable, n_jobs=1, verbose=0):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(item) for item in iterable)


def assembly_align_to_fps(output_assembly, fps, mode="portion"):
    EPS = 1e-6  # prevent the duration from being slightly larger than the last time bin
    interval = 1000 / fps
    duration = output_assembly["time_bin_end"].values[-1]
    target_time_bin_starts = np.arange(0, duration-EPS, interval)
    target_time_bin_ends = target_time_bin_starts + interval
    original_time_bin_starts = output_assembly["time_bin_start"].values
    if len(original_time_bin_starts) == len(target_time_bin_starts) and\
        np.isclose(original_time_bin_starts, target_time_bin_starts).all():
        return output_assembly
    else:
        target_time_bins = [(start, end) for start, end in zip(target_time_bin_starts, target_time_bin_ends)]
        return assembly_time_align(output_assembly, target_time_bins, mode=mode)
