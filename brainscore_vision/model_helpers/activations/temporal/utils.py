import numpy as np

from brainscore_vision.model_helpers.brain_transformation.temporal import assembly_time_align


def parallelize(func, iterable, n_jobs=1, verbose=0):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(item) for item in iterable)


def assembly_align_to_fps(output_assembly, fps, mode="portion"):
    EPS = 1e-9  # prevent the duration from being slightly larger than the last time bin
    interval = 1000 / fps
    duration = output_assembly["time_bin_end"].values[-1]
    target_time_bin_starts = np.arange(0, duration-EPS, interval)
    target_time_bin_ends = target_time_bin_starts + interval
    target_time_bins = [(start, end) for start, end in zip(target_time_bin_starts, target_time_bin_ends)]
    return assembly_time_align(output_assembly, target_time_bins, mode=mode)


# get the inferencer from any model
def get_inferencer(any_model):
    from brainscore_vision.model_helpers.activations.temporal.core.extractor import Inferencer, ActivationsExtractor
    from brainscore_vision.model_helpers.activations.temporal.model.base import ActivationWrapper
    from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

    if isinstance(any_model, Inferencer): return any_model
    if isinstance(any_model, ActivationWrapper): return any_model._extractor.inferencer
    if isinstance(any_model, ModelCommitment): return any_model.activations_model._extractor.inferencer
    if isinstance(any_model, ActivationsExtractor): return any_model.inferencer
    raise ValueError(f"Cannot find inferencer from the model {any_model}")

def get_base_model(any_model):
    from brainscore_vision.model_helpers.activations.temporal.model.base import ActivationWrapper
    from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

    if isinstance(any_model, ActivationWrapper): return any_model
    if isinstance(any_model, ModelCommitment): return any_model.activations_model
    raise ValueError(f"Cannot find inferencer from the model {any_model}")


# get the layers from the layer_activation_format
def get_specified_layers(any_model):
    inferencer = get_inferencer(any_model)
    return list(inferencer.layer_activation_format.keys())

# switch the inferencer at any level
# specify key='same' to retrive the same parameter from the original inferencer
def switch_inferencer(any_model, new_inferencer_cls, **kwargs):
    inferencer = get_inferencer(any_model)
    base_model = get_base_model(any_model)
    for k, v in kwargs.items():
        if v == 'same': kwargs[k] = getattr(inferencer, k)
    base_model.build_extractor(new_inferencer_cls, **kwargs)