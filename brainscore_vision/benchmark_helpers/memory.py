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

from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, TrainTestNeuralBenchmark, RSABenchmark, timebins_from_assembly
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.model_interface import BrainModel

_logger = logging.getLogger(__name__)

# Default path for the persistent calibration table.
# Prefer the file bundled with the package; fall back to the user-local path
# so that a local calibration run (mem_profile_suite.py --calibrate) can
# extend or override the shipped table without touching the source tree.
_BUNDLED_CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), 'benchmark_costs.json')
_DEFAULT_CALIBRATION_PATH = (
    _BUNDLED_CALIBRATION_PATH
    if os.path.exists(_BUNDLED_CALIBRATION_PATH)
    else os.path.expanduser('~/.brainscore/benchmark_costs.json')
)

# float32 = 4 bytes per element
_BYTES_PER_ELEMENT = 4

# Overhead multiplier on top of the activation assembly size.
# Accounts for xarray coordinate arrays, regression/CV matrices, and
# temporary buffers.  Calibrated against MajajHong2015.IT-pls (resnet50,
# no PCA): 1.91 GB assembly → 9.98 GB observed peak delta → 5.2× real
# overhead.  Using 6× to stay slightly conservative.
_OVERHEAD_FACTOR = 6

# Overhead multiplier applied to the activation array for PLS benchmarks.
# PLS regression builds cross-covariance matrices of shape
# (num_features × num_neuroids) whose memory scales with the model's feature
# count.  The calibrated fixed_benchmark_cost is therefore NOT model-independent
# for PLS — it was measured on alexnet (~9K features) and severely underestimates
# for large-feature models (200K+ features).
#
# Formula for PLS:  total = activation_gb × _PLS_OVERHEAD_FACTOR + fixed_cost_gb
#   where fixed_cost_gb covers the neural-assembly side (truly model-independent).
#
# Validated against a 3-model × 2-PLS-benchmark grid:
#   worst miss after fix: resnet50 × Cadena2017-pls  →  -12.7%  (within 15%)
_PLS_OVERHEAD_FACTOR = 7


@dataclass
class MemoryEstimate:
    """Breakdown of the estimated memory footprint for a benchmark run."""
    num_stimuli: int
    num_trials: int
    num_features: int
    num_timebins: int
    activation_gb: float        # activation array only
    total_estimated_gb: float   # see formula description below
    available_gb: float
    fixed_benchmark_cost_gb: Optional[float] = None  # None → overhead-factor fallback was used
    is_pls: bool = False        # True → PLS formula was used (activation × _PLS_OVERHEAD_FACTOR + fixed_cost)
    # formula_type: 'pls' | 'rdm' | 'ridge_formula' | 'calibrated' | 'fallback'
    formula_type: str = 'fallback'
    rdm_overhead_gb: Optional[float] = None  # n_stimuli^2 term used in RDM and ridge-formula paths

    @property
    def will_oom(self) -> bool:
        return self.total_estimated_gb > self.available_gb

    def __str__(self) -> str:
        status = "OOM LIKELY" if self.will_oom else "OK"
        if self.formula_type == 'pls':
            fixed_str = (f" + {self.fixed_benchmark_cost_gb:.2f} GB fixed cost"
                         if self.fixed_benchmark_cost_gb else "")
            formula = (f"{self.activation_gb:.2f} GB activations "
                       f"×{_PLS_OVERHEAD_FACTOR} (PLS){fixed_str}")
        elif self.formula_type == 'rdm':
            formula = (f"{self.activation_gb:.2f} GB activations "
                       f"×3 (RDM pairwise distance overhead → {self.total_estimated_gb:.1f} GB total)")
        elif self.formula_type == 'ridge_large_feature':
            formula = (f"{self.activation_gb:.2f} GB activations "
                       f"×{_OVERHEAD_FACTOR} (ridge SVD path: n_features > n_stimuli → {self.total_estimated_gb:.1f} GB total)")
        elif self.formula_type == 'ridge_formula':
            formula = (f"{self.activation_gb:.2f} GB activations "
                       f"+ {self.rdm_overhead_gb:.2f} GB gram matrix ({self.num_stimuli}²×4B)")
        elif self.formula_type == 'calibrated':
            formula = (f"{self.activation_gb:.2f} GB activations "
                       f"+ {self.fixed_benchmark_cost_gb:.2f} GB fixed benchmark cost (calibrated)")
        else:
            formula = (f"{self.activation_gb:.2f} GB "
                       f"(×{_OVERHEAD_FACTOR} overhead → {self.total_estimated_gb:.1f} GB total)")
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


def _is_pls_benchmark(benchmark) -> bool:
    """Return True if the benchmark uses PLS regression.

    PLS cross-covariance matrices scale with num_features, so the calibrated
    fixed_benchmark_cost (measured on alexnet with ~9K features) does not
    generalise to large-feature models.  A dedicated PLS overhead formula is
    applied instead.  Detection is based on the naming convention: all PLS
    benchmarks in brainscore_vision end with ``-pls`` or ``-reverse_pls``.
    """
    ident = str(getattr(benchmark, 'identifier', ''))
    return ident.endswith('-pls') or ident.endswith('-reverse_pls') or '-temporal-pls' in ident


def _is_rdm_benchmark(benchmark) -> bool:
    """Return True if the benchmark uses RDM/RSA.

    RDM overhead is n_stimuli^2 × 4B — completely model-independent.
    Detected via the ``-rdm`` suffix or RSABenchmark instance type.
    """
    if isinstance(benchmark, RSABenchmark):
        return True
    return str(getattr(benchmark, 'identifier', '')).endswith('-rdm')


def _is_ridge_benchmark(benchmark) -> bool:
    """Return True if the benchmark uses ridge or ridgecv regression.

    The gram matrix for ridge is n_stimuli × n_stimuli — model-independent —
    so we can compute a formula-based estimate when no calibration entry exists.
    """
    ident = str(getattr(benchmark, 'identifier', ''))
    return ident.endswith('-ridge') or ident.endswith('-ridgecv')


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

    elif isinstance(benchmark, RSABenchmark):
        stimulus_set = benchmark._assembly.stimulus_set
        num_stimuli = int(stimulus_set['stimulus_id'].nunique())
        num_trials = benchmark._number_of_trials
        timebins = timebins_from_assembly(benchmark._assembly)
        region = benchmark.region
        visual_degrees = benchmark._visual_degrees

    else:
        raise TypeError(
            f"preallocate_memory supports NeuralBenchmark, TrainTestNeuralBenchmark, and RSABenchmark; "
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

    # ------------------------------------------------------------------ #
    #  Choose the right formula based on the benchmark's regression type  #
    #                                                                     #
    #  PLS: cross-covariance matrices scale with num_features — use      #
    #  activation × _PLS_OVERHEAD_FACTOR.  This is approximate; a       #
    #  warning is printed.                                               #
    #                                                                     #
    #  RDM/RSA: pairwise distance computation passes through the full    #
    #  activation matrix — overhead ≈ 2× activation_gb.  Use 3× total.  #
    #                                                                     #
    #  Ridge/RidgeCV — two regimes depending on feature count:           #
    #                                                                     #
    #    n_features ≤ n_stimuli (primal solver): calibrated fixed cost   #
    #    is accurate — gram matrix is n_stimuli×n_stimuli and is model-  #
    #    independent.                                                     #
    #                                                                     #
    #    n_features > n_stimuli (sklearn switches to SVD of X): overhead #
    #    ≈ 5× activation_gb — SVD creates V^T (same shape as X) and     #
    #    U (n_stimuli×n_stimuli), so total ≈ 6× activation_gb.  The     #
    #    calibrated fixed cost was measured on a small model (alexnet,   #
    #    n_features < n_stimuli for most benchmarks) and severely        #
    #    underestimates in this regime.  Use the ×6 fallback instead so  #
    #    the pre-flight raises MemoryError cleanly before the OS kills   #
    #    the container with no Python traceback.                          #
    # ------------------------------------------------------------------ #
    is_pls = _is_pls_benchmark(benchmark)
    is_rdm = _is_rdm_benchmark(benchmark)
    is_ridge = _is_ridge_benchmark(benchmark)
    ridge_large_feature = is_ridge and num_features > num_stimuli

    rdm_overhead_gb = None
    if is_pls:
        total_estimated_gb = activation_gb * _PLS_OVERHEAD_FACTOR + (fixed_benchmark_cost_gb or 0.0)
        formula_type = 'pls'
    elif is_rdm:
        # Overhead ≈ 2× activation_gb (scales with features, not n_stimuli²).
        # Validated across alexnet/resnet50/ViT on Allen2022_fmri.IT-rdm.
        rdm_overhead_gb = 2 * activation_gb
        total_estimated_gb = activation_gb + rdm_overhead_gb  # = 3 × activation_gb
        formula_type = 'rdm'
    elif ridge_large_feature:
        # n_features > n_stimuli: sklearn SVD path — overhead ≈ 5× activation_gb.
        # Validated: resnet50/ViT × Gifford2022.IT-ridgecv both gave exactly 5.1×.
        # Use ×6 total (activation + 5× overhead) to stay conservative and ensure
        # the pre-flight MemoryError fires before the OS kills the container.
        total_estimated_gb = activation_gb * _OVERHEAD_FACTOR
        formula_type = 'ridge_large_feature'
    elif is_ridge and fixed_benchmark_cost_gb is not None:
        total_estimated_gb = activation_gb + fixed_benchmark_cost_gb
        formula_type = 'calibrated'
    elif is_ridge:
        # No calibration entry, primal regime: gram matrix is n_stimuli×n_stimuli
        rdm_overhead_gb = (num_stimuli ** 2) * _BYTES_PER_ELEMENT / (1024 ** 3)
        total_estimated_gb = activation_gb + rdm_overhead_gb
        formula_type = 'ridge_formula'
    elif fixed_benchmark_cost_gb is not None:
        total_estimated_gb = activation_gb + fixed_benchmark_cost_gb
        formula_type = 'calibrated'
    else:
        total_estimated_gb = activation_gb * _OVERHEAD_FACTOR
        formula_type = 'fallback'

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
        is_pls=is_pls,
        formula_type=formula_type,
        rdm_overhead_gb=rdm_overhead_gb,
    )

    _logger.info(str(estimate))
    verdict = "OOM LIKELY" if estimate.will_oom else "OK"
    print(
        f"[pre-flight] [{verdict}]  "
        f"{estimate.total_estimated_gb:.2f} GB needed  /  {estimate.available_gb:.1f} GB available  "
        f"[{formula_type}]\n"
        f"  {estimate.num_stimuli:,} stimuli  ×  {estimate.num_features:,} features  ×  "
        f"{estimate.num_timebins} timebins  =  {estimate.activation_gb:.3f} GB activation",
        end='',
        flush=True,
    )
    if formula_type == 'pls':
        fixed_str = (f"  +  {estimate.fixed_benchmark_cost_gb:.3f} GB fixed cost"
                     if estimate.fixed_benchmark_cost_gb is not None else "")
        print(f"  ×{_PLS_OVERHEAD_FACTOR} (PLS){fixed_str}  =  {estimate.total_estimated_gb:.3f} GB total",
              flush=True)
        print(
            f"[pre-flight] WARNING: PLS overhead multiplier (×{_PLS_OVERHEAD_FACTOR}) is approximate. "
            f"Actual usage can vary significantly depending on model feature count and convergence.",
            flush=True,
        )
    elif formula_type == 'ridge_large_feature':
        print(f"  ×{_OVERHEAD_FACTOR} (ridge SVD: n_features={num_features:,} > n_stimuli={num_stimuli:,})"
              f"  =  {estimate.total_estimated_gb:.3f} GB total", flush=True)
    elif formula_type == 'rdm':
        print(f"  ×3 (RDM pairwise overhead)"
              f"  =  {estimate.total_estimated_gb:.3f} GB total", flush=True)
    elif formula_type == 'ridge_formula':
        print(f"  +  {estimate.rdm_overhead_gb:.3f} GB gram matrix ({num_stimuli:,}²×4B)  "
              f"[no calibration entry — formula estimate]"
              f"  =  {estimate.total_estimated_gb:.3f} GB total", flush=True)
    elif formula_type == 'calibrated':
        print(f"  +  {estimate.fixed_benchmark_cost_gb:.3f} GB benchmark overhead (calibrated)"
              f"  =  {estimate.total_estimated_gb:.3f} GB total", flush=True)
    else:
        print(f"  ×{_OVERHEAD_FACTOR}  =  {estimate.total_estimated_gb:.3f} GB total", flush=True)

    # Structured sentinel for CloudWatch Insights calibration queries and reliable
    # OOM signal parsing by the scoring orchestrator. Every pre-flight run emits
    # this line regardless of outcome — filter on will_oom=true for OOM cases.
    # Query example: filter @message like "BRAINSCORE_PREFLIGHT"
    #                | stats avg(estimate_gb) by benchmark_id, formula_type
    print(
        f"BRAINSCORE_PREFLIGHT {json.dumps({'estimate_gb': round(total_estimated_gb, 3), 'available_gb': round(available_gb, 1), 'formula_type': formula_type, 'will_oom': estimate.will_oom, 'num_features': num_features, 'num_stimuli': num_stimuli})}",
        flush=True,
    )

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
