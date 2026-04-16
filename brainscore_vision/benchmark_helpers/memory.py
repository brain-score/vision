"""
Memory estimation utilities for Brain-Score benchmarks.

Call :func:`preallocate_memory` before scoring to detect OOM errors early,
rather than discovering them 6+ hours into a benchmark run.

Example usage::

    from brainscore_vision import load_model, load_benchmark
    from brainscore_vision.benchmark_helpers.memory import preallocate_memory

    model = load_model('resnet50')
    benchmark = load_benchmark('MajajHong2015public.IT-pls')
    estimate = preallocate_memory(model, benchmark)   # raises MemoryError if OOM
    score = benchmark(model)
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import psutil

from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, TrainTestNeuralBenchmark
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.model_interface import BrainModel

_logger = logging.getLogger(__name__)

# Default path for the persistent calibration table
_DEFAULT_CALIBRATION_PATH = os.path.expanduser('~/.brainscore/benchmark_costs.json')

# float32 = 4 bytes per element
_BYTES_PER_ELEMENT = 4

# Overhead multiplier on top of the activation assembly size.
# Accounts for xarray coordinate arrays, regression/CV matrices, and
# temporary buffers.  Calibrated against MajajHong2015.IT-pls (resnet50,
# no PCA): 1.91 GB assembly → 9.98 GB observed peak delta → 5.2× real
# overhead.  Using 6× to stay slightly conservative.
_OVERHEAD_FACTOR = 6


@dataclass
class MemoryEstimate:
    """Breakdown of the estimated memory footprint for a benchmark run."""
    num_stimuli: int
    num_trials: int
    num_features: int
    num_timebins: int
    activation_gb: float        # activation array only
    total_estimated_gb: float   # activation_gb + fixed_benchmark_cost_gb, or activation_gb × overhead
    available_gb: float
    fixed_benchmark_cost_gb: Optional[float] = None  # None → overhead-factor fallback was used

    @property
    def will_oom(self) -> bool:
        return self.total_estimated_gb > self.available_gb

    def __str__(self) -> str:
        status = "OOM LIKELY" if self.will_oom else "OK"
        if self.fixed_benchmark_cost_gb is not None:
            formula = (
                f"{self.activation_gb:.2f} GB activations "
                f"+ {self.fixed_benchmark_cost_gb:.2f} GB fixed benchmark cost"
            )
        else:
            formula = (
                f"{self.activation_gb:.2f} GB  "
                f"(×{_OVERHEAD_FACTOR} overhead → {self.total_estimated_gb:.1f} GB total)"
            )
        return (
            f"[{status}] Memory estimate: {self.total_estimated_gb:.1f} GB needed, "
            f"{self.available_gb:.1f} GB available\n"
            f"  Activations: {self.num_stimuli} stimuli × {self.num_features:,} features "
            f"× {self.num_timebins} timebins = {formula}"
        )


def load_calibration(path: Optional[str] = None) -> dict:
    """Load the benchmark fixed-cost table from disk.

    Returns an empty dict if the file does not exist yet.
    The file is written by :func:`save_calibration` (or by the
    ``--calibrate`` mode of ``mem_profile_suite.py``).
    """
    path = path or _DEFAULT_CALIBRATION_PATH
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        _logger.warning(f"Could not load calibration from {path}: {e}")
        return {}


def save_calibration(costs: dict, path: Optional[str] = None) -> None:
    """Persist benchmark fixed costs to disk.

    Parameters
    ----------
    costs : dict
        ``{benchmark_identifier: fixed_cost_gb}`` mapping produced by a
        calibration run.
    path : str, optional
        Destination file.  Defaults to ``~/.brainscore/benchmark_costs.json``.
    """
    path = path or _DEFAULT_CALIBRATION_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(costs, f, indent=2, sort_keys=True)
    _logger.info(f"Calibration saved → {path}  ({len(costs)} benchmarks)")


def _get_probe_layer(model):
    """
    Return the committed layer string for the model's primary recording region,
    or None if it cannot be determined without triggering expensive layer selection.
    """
    try:
        # Navigate ModelCommitment → TemporalAligned → LayerMappedModel
        lm = getattr(model, 'layer_model', None)
        if lm is not None and hasattr(lm, '_layer_model'):
            lm = lm._layer_model  # TemporalAligned → LayerMappedModel
        if lm is None:
            lm = model  # might itself be LayerMappedModel-like

        rmap = getattr(lm, 'region_layer_map', None)
        if rmap is None:
            return None

        # Prefer IT, then any committed region.
        # Use dict.__contains__ to avoid triggering lazy RegionLayerMap.__getitem__
        for candidate_region in ['IT', 'V4', 'V2', 'V1']:
            if dict.__contains__(rmap, candidate_region):
                layers = dict.__getitem__(rmap, candidate_region)
                if layers is not None:
                    if isinstance(layers, (list, tuple)):
                        return layers[0]
                    return layers

        # If it's a plain dict (not lazy RegionLayerMap), just grab any value
        if type(rmap) is dict and rmap:
            return next(iter(rmap.values()))

    except Exception:
        pass
    return None


def preallocate_memory(
    model: BrainModel,
    benchmark,
    raise_if_oom: bool = True,
    fixed_benchmark_cost_gb: Optional[float] = None,
) -> Optional[MemoryEstimate]:
    """
    Estimate memory requirements before running a full benchmark.

    Probes the model with a single stimulus to get the actual feature count.
    The probe calls the activations extractor's ``_from_paths`` directly,
    bypassing ``from_stimulus_set`` / ``attach_stimulus_set_meta`` so that
    the probe cannot interfere with the subsequent scoring run's result cache.

    Estimates total memory as
    ``num_stimuli × num_features × num_timebins × 4 bytes × overhead``.

    num_trials is intentionally excluded: deterministic models process each
    unique stimulus once; the trial dimension in the neural assembly does not
    scale model memory.

    Parameters
    ----------
    model : BrainModel
        The candidate model that will be scored.
    benchmark : NeuralBenchmark or TrainTestNeuralBenchmark
        The benchmark the model will be scored on.
    raise_if_oom : bool, optional
        If ``True`` (default), raises :exc:`MemoryError` when the estimate
        exceeds available RAM.  If ``False``, logs a warning instead.

    Returns
    -------
    MemoryEstimate
        Estimated memory breakdown with a ``.will_oom`` property.

    Raises
    ------
    TypeError
        If *benchmark* is not a supported neural benchmark type.
    MemoryError
        If ``raise_if_oom=True`` and estimated memory exceeds available RAM.
    """
    if os.environ.get('BRAINSCORE_SKIP_MEMORY_CHECK', '0') == '1':
        _logger.debug("BRAINSCORE_SKIP_MEMORY_CHECK is set — skipping memory pre-check.")
        return None

    # ------------------------------------------------------------------ #
    #  1. Extract metadata from the benchmark                             #
    # ------------------------------------------------------------------ #
    if isinstance(benchmark, NeuralBenchmark):
        stimulus_set = benchmark._assembly.stimulus_set
        num_stimuli = int(stimulus_set['stimulus_id'].nunique())
        num_trials = benchmark._number_of_trials
        timebins = benchmark.timebins
        region = benchmark.region
        visual_degrees = benchmark._visual_degrees

    elif isinstance(benchmark, TrainTestNeuralBenchmark):
        train_ss = benchmark.train_assembly.stimulus_set
        test_ss = benchmark.test_assembly.stimulus_set
        stimulus_set = train_ss
        num_stimuli = int(train_ss['stimulus_id'].nunique()) + int(test_ss['stimulus_id'].nunique())
        num_trials = benchmark._number_of_trials
        timebins = benchmark.timebins
        region = benchmark.region
        visual_degrees = benchmark._visual_degrees

    else:
        raise TypeError(
            f"preallocate_memory supports NeuralBenchmark and TrainTestNeuralBenchmark; "
            f"got {type(benchmark).__name__}"
        )

    # ------------------------------------------------------------------ #
    #  2. Prepare probe stimulus (1 image, visual-degree corrected)       #
    # ------------------------------------------------------------------ #
    probe_set = stimulus_set.iloc[:1]
    probe_set.identifier = None
    probe_set = place_on_screen(
        probe_set,
        target_visual_degrees=model.visual_degrees(),
        source_visual_degrees=visual_degrees,
    )
    probe_stimulus_id = probe_set['stimulus_id'].values[0]
    probe_path = str(probe_set.get_stimulus(probe_stimulus_id))

    # ------------------------------------------------------------------ #
    #  3. Probe the model with 1 stimulus                                 #
    #                                                                     #
    #  We call _from_paths directly — bypassing from_stimulus_set and    #
    #  attach_stimulus_set_meta — so the probe cannot corrupt the        #
    #  activations cache used by the subsequent scoring run.             #
    #                                                                     #
    #  We do NOT disable LayerPCA: the probe should measure the feature  #
    #  count exactly as the scoring run will accumulate it (i.e. after   #
    #  PCA reduction when PCA is hooked, raw otherwise).                 #
    # ------------------------------------------------------------------ #
    _am = getattr(model, 'activations_model', None)
    _extractor = getattr(_am, '_extractor', None) if _am else None
    probe_layer = _get_probe_layer(model) if _extractor is not None else None

    if _extractor is not None and probe_layer is not None:
        # Fast path: call _from_paths directly — no attach_stimulus_set_meta
        probe_output = _extractor._from_paths(layers=[probe_layer], stimuli_paths=[probe_path])
        num_features = probe_output.sizes['neuroid']
        num_timebins = len(timebins)  # _from_paths has no time expansion; timebins from benchmark
    else:
        # Fallback: use the standard look_at pipeline
        model.start_recording(region, time_bins=timebins)
        probe_output = model.look_at(probe_set, number_of_trials=1)
        num_features = probe_output.sizes['neuroid']
        num_timebins = probe_output.sizes.get('time_bin', 1)

    _logger.info(
        f"Memory probe: benchmark={benchmark.identifier} region={region} "
        f"stimuli={num_stimuli} features={num_features} timebins={num_timebins}"
    )

    # ------------------------------------------------------------------ #
    #  4. Compute estimate and check against available RAM                #
    #                                                                     #
    #  num_trials excluded: deterministic models process each unique      #
    #  stimulus once; trial repetition does not scale model memory.       #
    # ------------------------------------------------------------------ #
    activation_bytes = num_stimuli * num_features * num_timebins * _BYTES_PER_ELEMENT
    activation_gb = activation_bytes / (1024 ** 3)

    # Auto-load from the calibration table if no explicit value was given
    if fixed_benchmark_cost_gb is None:
        _cal = load_calibration()
        fixed_benchmark_cost_gb = _cal.get(benchmark.identifier)
        if fixed_benchmark_cost_gb is not None:
            _logger.debug(
                f"Using calibrated fixed cost for {benchmark.identifier}: "
                f"{fixed_benchmark_cost_gb:.3f} GB"
            )

    if fixed_benchmark_cost_gb is not None:
        total_estimated_gb = activation_gb + fixed_benchmark_cost_gb
    else:
        total_estimated_gb = activation_gb * _OVERHEAD_FACTOR
    available_gb = psutil.virtual_memory().available / (1024 ** 3)

    estimate = MemoryEstimate(
        num_stimuli=num_stimuli,
        num_trials=num_trials,
        num_features=num_features,
        num_timebins=num_timebins,
        activation_gb=activation_gb,
        total_estimated_gb=total_estimated_gb,
        available_gb=available_gb,
        fixed_benchmark_cost_gb=fixed_benchmark_cost_gb,
    )

    _logger.info(str(estimate))
    verdict = "OOM LIKELY" if estimate.will_oom else "OK"
    formula_used = "calibrated" if estimate.fixed_benchmark_cost_gb is not None else f"×{_OVERHEAD_FACTOR} fallback"
    print(
        f"[pre-flight] [{verdict}]  "
        f"{estimate.total_estimated_gb:.2f} GB needed  /  {estimate.available_gb:.1f} GB available  "
        f"[{formula_used}]\n"
        f"  {estimate.num_stimuli:,} stimuli  ×  {estimate.num_features:,} features  ×  "
        f"{estimate.num_timebins} timebins  =  {estimate.activation_gb:.3f} GB activation",
        end='',
        flush=True,
    )
    if estimate.fixed_benchmark_cost_gb is not None:
        print(f"  +  {estimate.fixed_benchmark_cost_gb:.3f} GB benchmark overhead  "
              f"=  {estimate.total_estimated_gb:.3f} GB total", flush=True)
    else:
        print(f"  ×{_OVERHEAD_FACTOR}  =  {estimate.total_estimated_gb:.3f} GB total", flush=True)

    if estimate.will_oom:
        msg = (
            f"preallocate_memory: {str(estimate)}. "
            f"Consider reducing layer output dimensionality (e.g. via LayerPCA), "
            f"running on a machine with more RAM, or selecting a different layer."
        )
        if raise_if_oom:
            raise MemoryError(msg)
        else:
            _logger.warning(msg)

    return estimate
