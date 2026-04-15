"""
Memory Profile Suite  —  5 × 5 estimate vs. actual
====================================================
For every (model, benchmark) pair:
  1. Run the pre-flight probe  →  estimated peak GB
  2. Run the full benchmark    →  actual peak RSS delta
  3. Compare estimate to actual and report accuracy

Usage
-----
    python scripts/mem_profile_suite.py [--csv out.csv] [--skip-score]

--skip-score  runs only the pre-flight probes (no actual scoring).
"""
import os
import sys
import time
import argparse
import csv
import logging
import threading

# ---------------------------------------------------------------------------
# Resolve local repos so the script works without installation
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_vision_root = os.path.dirname(_script_dir)
_core_root = os.path.join(os.path.dirname(_vision_root), 'core')
for _p in [_vision_root, _core_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.WARNING)

print("Importing brainscore_vision... ", end='', flush=True)
import brainscore_vision  # noqa: E402
print("done.", flush=True)

import psutil  # noqa: E402

# ---------------------------------------------------------------------------
# Model / benchmark lists
# ---------------------------------------------------------------------------
MODELS = [
    'resnet50_tutorial',
    'alexnet',
    'vit_large_patch14_clip_224:openai_ft_in1k',
    'VOneCORnet-S',
    'efficientnet_b0',
]

BENCHMARKS = [
    'MajajHong2015.IT-pls',
    'Sanghavi2020.IT-pls',
    'Papale2025.IT-ridgecv',
    'Hebart2023_fmri.V4-ridgecv',
    'Allen2022_fmri.IT-ridge',
]

_BM_SHORT = {
    'MajajHong2015.IT-pls':        'MajajHong.IT',
    'Sanghavi2020.IT-pls':         'Sanghavi.IT',
    'Papale2025.IT-ridgecv':       'Papale25.IT',
    'Hebart2023_fmri.V4-ridgecv':  'Hebart23.V4',
    'Allen2022_fmri.IT-ridge':     'Allen22.IT',
}

# All registered leaf benchmarks — used by --calibrate mode.
ALL_BENCHMARKS = [
    # Allen2022 fMRI (volumetric)
    'Allen2022_fmri.V1-ridge', 'Allen2022_fmri.V2-ridge',
    'Allen2022_fmri.V4-ridge', 'Allen2022_fmri.IT-ridge',
    'Allen2022_fmri.V1-rdm',   'Allen2022_fmri.V2-rdm',
    'Allen2022_fmri.V4-rdm',   'Allen2022_fmri.IT-rdm',
    'Allen2022_fmri_4subj.V1-ridge', 'Allen2022_fmri_4subj.V2-ridge',
    'Allen2022_fmri_4subj.V4-ridge', 'Allen2022_fmri_4subj.IT-ridge',
    'Allen2022_fmri_4subj.V1-rdm',   'Allen2022_fmri_4subj.V2-rdm',
    'Allen2022_fmri_4subj.V4-rdm',   'Allen2022_fmri_4subj.IT-rdm',
    # Allen2022 fMRI (surface)
    'Allen2022_fmri_surface.V1-ridge', 'Allen2022_fmri_surface.V2-ridge',
    'Allen2022_fmri_surface.V4-ridge', 'Allen2022_fmri_surface.IT-ridge',
    'Allen2022_fmri_surface.V1-rdm',   'Allen2022_fmri_surface.V2-rdm',
    'Allen2022_fmri_surface.V4-rdm',   'Allen2022_fmri_surface.IT-rdm',
    'Allen2022_fmri_surface_4subj.V1-ridge', 'Allen2022_fmri_surface_4subj.V2-ridge',
    'Allen2022_fmri_surface_4subj.V4-ridge', 'Allen2022_fmri_surface_4subj.IT-ridge',
    'Allen2022_fmri_surface_4subj.V1-rdm',   'Allen2022_fmri_surface_4subj.V2-rdm',
    'Allen2022_fmri_surface_4subj.V4-rdm',   'Allen2022_fmri_surface_4subj.IT-rdm',
    # Baker2022
    'Baker2022frankenstein-accuracy_delta',
    'Baker2022fragmented-accuracy_delta',
    'Baker2022inverted-accuracy_delta',
    # BMD2024
    'BMD2024.texture_1Behavioral-accuracy_distance',
    'BMD2024.texture_2Behavioral-accuracy_distance',
    'BMD2024.dotted_1Behavioral-accuracy_distance',
    'BMD2024.dotted_2Behavioral-accuracy_distance',
    # Bracci2019
    'Bracci2019.anteriorVTC-rdm',
    # Cadena2017
    'Cadena2017-pls', 'Cadena2017-mask',
    # Coggan2024
    'tong.Coggan2024_fMRI.V1-rdm', 'tong.Coggan2024_fMRI.V2-rdm',
    'tong.Coggan2024_fMRI.V4-rdm', 'tong.Coggan2024_fMRI.IT-rdm',
    'tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity',
    # Ferguson2024
    'Ferguson2024circle_line-value_delta', 'Ferguson2024color-value_delta',
    'Ferguson2024convergence-value_delta',  'Ferguson2024eighth-value_delta',
    'Ferguson2024gray_easy-value_delta',    'Ferguson2024gray_hard-value_delta',
    'Ferguson2024half-value_delta',         'Ferguson2024juncture-value_delta',
    'Ferguson2024lle-value_delta',          'Ferguson2024llh-value_delta',
    'Ferguson2024quarter-value_delta',      'Ferguson2024round_f-value_delta',
    'Ferguson2024round_v-value_delta',      'Ferguson2024tilted_line-value_delta',
    # FreemanZiemba2013
    'FreemanZiemba2013.V1-pls',       'FreemanZiemba2013.V2-pls',
    'FreemanZiemba2013public.V1-pls', 'FreemanZiemba2013public.V2-pls',
    # Geirhos2021
    'Geirhos2021colour-top1',              'Geirhos2021colour-error_consistency',
    'Geirhos2021contrast-top1',            'Geirhos2021contrast-error_consistency',
    'Geirhos2021cueconflict-top1',         'Geirhos2021cueconflict-error_consistency',
    'Geirhos2021edge-top1',                'Geirhos2021edge-error_consistency',
    'Geirhos2021eidolonI-top1',            'Geirhos2021eidolonI-error_consistency',
    'Geirhos2021eidolonII-top1',           'Geirhos2021eidolonII-error_consistency',
    'Geirhos2021eidolonIII-top1',          'Geirhos2021eidolonIII-error_consistency',
    'Geirhos2021falsecolour-top1',         'Geirhos2021falsecolour-error_consistency',
    'Geirhos2021highpass-top1',            'Geirhos2021highpass-error_consistency',
    'Geirhos2021lowpass-top1',             'Geirhos2021lowpass-error_consistency',
    'Geirhos2021phasescrambling-top1',     'Geirhos2021phasescrambling-error_consistency',
    'Geirhos2021powerequalisation-top1',   'Geirhos2021powerequalisation-error_consistency',
    'Geirhos2021rotation-top1',            'Geirhos2021rotation-error_consistency',
    'Geirhos2021silhouette-top1',          'Geirhos2021silhouette-error_consistency',
    'Geirhos2021sketch-top1',              'Geirhos2021sketch-error_consistency',
    'Geirhos2021stylized-top1',            'Geirhos2021stylized-error_consistency',
    'Geirhos2021uniformnoise-top1',        'Geirhos2021uniformnoise-error_consistency',
    # Gifford2022
    'Gifford2022.IT-ridge', 'Gifford2022.IT-ridgecv',
    # Hebart2023
    'Hebart2023-match',
    'Hebart2023_fmri.V1-ridge',   'Hebart2023_fmri.V2-ridge',
    'Hebart2023_fmri.V4-ridge',   'Hebart2023_fmri.IT-ridge',
    'Hebart2023_fmri.V1-ridgecv', 'Hebart2023_fmri.V2-ridgecv',
    'Hebart2023_fmri.V4-ridgecv', 'Hebart2023_fmri.IT-ridgecv',
    # Hermann2020
    'Hermann2020cueconflict-shape_bias', 'Hermann2020cueconflict-shape_match',
    # Igustibagus2024
    'Igustibagus2024-ridge', 'Igustibagus2024.IT_readout-accuracy',
    # ImageNet
    'ImageNet-top1',
    'ImageNet-C-noise-top1', 'ImageNet-C-blur-top1',
    'ImageNet-C-weather-top1', 'ImageNet-C-digital-top1',
    # Islam2021
    'Islam2021-shape_v1_dimensionality',  'Islam2021-texture_v1_dimensionality',
    'Islam2021-shape_v2_dimensionality',  'Islam2021-texture_v2_dimensionality',
    'Islam2021-shape_v4_dimensionality',  'Islam2021-texture_v4_dimensionality',
    'Islam2021-shape_it_dimensionality',  'Islam2021-texture_it_dimensionality',
    # Kar2019
    'Kar2019-ost',
    # Lonnqvist2024
    'Lonnqvist2024_InlabInstructionsBehavioralAccuracyDistance',
    'Lonnqvist2024_InlabNoInstructionsBehavioralAccuracyDistance',
    'Lonnqvist2024_OnlineNoInstructionsBehavioralAccuracyDistance',
    'Lonnqvist2024_EngineeringAccuracy',
    # MajajHong2015
    'MajajHong2015.V4-pls',              'MajajHong2015.IT-pls',
    'MajajHong2015public.V4-pls',        'MajajHong2015public.IT-pls',
    'MajajHong2015public.V4-temporal-pls', 'MajajHong2015public.IT-temporal-pls',
    'MajajHong2015public.V4-reverse_pls', 'MajajHong2015public.IT-reverse_pls',
    # Malania2007
    'Malania2007.short2-threshold_elevation',  'Malania2007.short4-threshold_elevation',
    'Malania2007.short6-threshold_elevation',  'Malania2007.short8-threshold_elevation',
    'Malania2007.short16-threshold_elevation', 'Malania2007.equal2-threshold_elevation',
    'Malania2007.long2-threshold_elevation',   'Malania2007.equal16-threshold_elevation',
    'Malania2007.long16-threshold_elevation',  'Malania2007.vernieracuity-threshold',
    # Maniquet2024
    'Maniquet2024-confusion_similarity', 'Maniquet2024-tasks_consistency',
    # Marques2020
    'Marques2020_Cavanaugh2002-grating_summation_field',
    'Marques2020_Cavanaugh2002-surround_diameter',
    'Marques2020_Cavanaugh2002-surround_suppression_index',
    'Marques2020_DeValois1982-pref_or',
    'Marques2020_DeValois1982-peak_sf',
    'Marques2020_FreemanZiemba2013-texture_modulation_index',
    'Marques2020_FreemanZiemba2013-abs_texture_modulation_index',
    'Marques2020_FreemanZiemba2013-texture_selectivity',
    'Marques2020_FreemanZiemba2013-texture_sparseness',
    'Marques2020_FreemanZiemba2013-texture_variance_ratio',
    'Marques2020_FreemanZiemba2013-max_texture',
    'Marques2020_FreemanZiemba2013-max_noise',
    'Marques2020_Ringach2002-circular_variance',  'Marques2020_Ringach2002-or_bandwidth',
    'Marques2020_Ringach2002-orth_pref_ratio',    'Marques2020_Ringach2002-or_selective',
    'Marques2020_Ringach2002-cv_bandwidth_ratio', 'Marques2020_Ringach2002-opr_cv_diff',
    'Marques2020_Ringach2002-max_dc',             'Marques2020_Ringach2002-modulation_ratio',
    'Marques2020_Schiller1976-sf_selective',      'Marques2020_Schiller1976-sf_bandwidth',
    # ObjectNet
    'ObjectNet-top1',
    # Papale2025
    'Papale2025.V1-ridge',   'Papale2025.V4-ridge',   'Papale2025.IT-ridge',
    'Papale2025.V1-ridgecv', 'Papale2025.V4-ridgecv', 'Papale2025.IT-ridgecv',
    # Rajalingham2018
    'Rajalingham2018-i2n', 'Rajalingham2018public-i2n',
    # Rajalingham2020
    'Rajalingham2020.IT-pls',
    # Sanghavi2020
    'Sanghavi2020.V4-pls',     'Sanghavi2020.IT-pls',
    'SanghaviJozwik2020.V4-pls', 'SanghaviJozwik2020.IT-pls',
    'SanghaviMurty2020.V4-pls',  'SanghaviMurty2020.IT-pls',
    # Scialom2024
    'Scialom2024_rgbBehavioralAccuracyDistance',
    'Scialom2024_contoursBehavioralAccuracyDistance',
    'Scialom2024_phosphenes-12BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-16BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-21BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-27BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-35BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-46BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-59BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-77BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-100BehavioralAccuracyDistance',
    'Scialom2024_segments-12BehavioralAccuracyDistance',
    'Scialom2024_segments-16BehavioralAccuracyDistance',
    'Scialom2024_segments-21BehavioralAccuracyDistance',
    'Scialom2024_segments-27BehavioralAccuracyDistance',
    'Scialom2024_segments-35BehavioralAccuracyDistance',
    'Scialom2024_segments-46BehavioralAccuracyDistance',
    'Scialom2024_segments-59BehavioralAccuracyDistance',
    'Scialom2024_segments-77BehavioralAccuracyDistance',
    'Scialom2024_segments-100BehavioralAccuracyDistance',
    'Scialom2024_phosphenes-allBehavioralErrorConsistency',
    'Scialom2024_segments-allBehavioralErrorConsistency',
    'Scialom2024_phosphenes-allBehavioralAccuracyDistance',
    'Scialom2024_segments-allBehavioralAccuracyDistance',
    'Scialom2024_rgbEngineeringAccuracy',
    'Scialom2024_contoursEngineeringAccuracy',
    'Scialom2024_phosphenes-12EngineeringAccuracy',
    'Scialom2024_phosphenes-16EngineeringAccuracy',
    'Scialom2024_phosphenes-21EngineeringAccuracy',
    'Scialom2024_phosphenes-27EngineeringAccuracy',
    'Scialom2024_phosphenes-35EngineeringAccuracy',
    'Scialom2024_phosphenes-46EngineeringAccuracy',
    'Scialom2024_phosphenes-59EngineeringAccuracy',
    'Scialom2024_phosphenes-77EngineeringAccuracy',
    'Scialom2024_phosphenes-100EngineeringAccuracy',
    'Scialom2024_segments-12EngineeringAccuracy',
    'Scialom2024_segments-16EngineeringAccuracy',
    'Scialom2024_segments-21EngineeringAccuracy',
    'Scialom2024_segments-27EngineeringAccuracy',
    'Scialom2024_segments-35EngineeringAccuracy',
    'Scialom2024_segments-46EngineeringAccuracy',
    'Scialom2024_segments-59EngineeringAccuracy',
    'Scialom2024_segments-77EngineeringAccuracy',
    'Scialom2024_segments-100EngineeringAccuracy',
]

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
_RESET  = '\033[0m'
_BOLD   = '\033[1m'
_GREEN  = '\033[32m'
_YELLOW = '\033[33m'
_RED    = '\033[31m'
_CYAN   = '\033[36m'
_DIM    = '\033[2m'
_BLUE   = '\033[34m'


def _c(text, colour):
    return f"{colour}{text}{_RESET}"


def _step(msg, indent=4):
    print(f"{' ' * indent}{_c('→', _BLUE)} {msg}", flush=True)


def _substep(msg, indent=6):
    print(f"{' ' * indent}{_c('·', _DIM)} {msg}", flush=True)


def _gb(n_bytes):
    return f"{n_bytes / (1024 ** 3):.2f} GB"


# ---------------------------------------------------------------------------
# Peak RSS monitor (background thread)
# ---------------------------------------------------------------------------

class _PeakMonitor:
    def __init__(self, interval=0.5):
        self._proc = psutil.Process(os.getpid())
        self._interval = interval
        self._peak = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self._peak

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except psutil.NoSuchProcess:
                break
            self._stop.wait(self._interval)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _compare_label(estimate_gb, actual_delta_gb):
    """Returns (colour, verdict_string, ratio)."""
    if actual_delta_gb <= 0.01:
        return _GREEN, "no measurable RSS delta", None
    ratio = estimate_gb / actual_delta_gb
    if ratio >= 0.8:
        return _GREEN,  f"ACCURATE  ({ratio:.2f}× of actual)", ratio
    elif ratio >= 0.4:
        under = actual_delta_gb - estimate_gb
        pct = (1 - ratio) * 100
        return _YELLOW, f"UNDER by {under:.2f} GB  ({pct:.0f}% under)", ratio
    else:
        under = actual_delta_gb - estimate_gb
        pct = (1 - ratio) * 100
        return _RED,    f"UNDER by {under:.2f} GB  ({pct:.0f}% under)", ratio


# ---------------------------------------------------------------------------
# Result helper
# ---------------------------------------------------------------------------

def _make_result(model_id, benchmark_id, **kw):
    return dict(model_id=model_id, benchmark_id=benchmark_id, **kw)


# ---------------------------------------------------------------------------
# Run one (model, benchmark) pair
# ---------------------------------------------------------------------------

def run_pair(model, model_id, benchmark, benchmark_id, skip_score=False):
    from brainscore_vision.benchmark_helpers.memory import preallocate_memory

    proc = psutil.Process(os.getpid())

    # ── Pre-flight probe ─────────────────────────────────────────────────
    _step("pre-flight probe  (1-stimulus forward pass)")
    t_probe = time.time()
    try:
        est = preallocate_memory(model, benchmark, raise_if_oom=False)
    except TypeError as e:
        _substep(_c(f"skipped — unsupported benchmark type: {e}", _DIM))
        return _make_result(model_id, benchmark_id, status='skip',
                            est_gb=None, act_gb=None, actual_delta_gb=None, score=None,
                            probe_elapsed=time.time() - t_probe,
                            score_elapsed=None, note=str(e)[:100])
    except Exception as e:
        _substep(_c(f"probe ERROR: {str(e)[:100]}", _RED))
        return _make_result(model_id, benchmark_id, status='error',
                            est_gb=None, act_gb=None, actual_delta_gb=None, score=None,
                            probe_elapsed=time.time() - t_probe,
                            score_elapsed=None, note=str(e)[:120])

    probe_elapsed = time.time() - t_probe

    if est is None:
        _substep(_c("skipped (BRAINSCORE_SKIP_MEMORY_CHECK set)", _DIM))
        return _make_result(model_id, benchmark_id, status='skip',
                            est_gb=None, act_gb=None, actual_delta_gb=None, score=None,
                            probe_elapsed=probe_elapsed, score_elapsed=None,
                            note='BRAINSCORE_SKIP_MEMORY_CHECK set')

    _substep(f"features={est.num_features:,}  stimuli={est.num_stimuli:,}  timebins={est.num_timebins}")
    _substep(
        f"activation = {est.activation_gb:.3f} GB  ×6 overhead  "
        f"→ {_c(f'estimate: {est.total_estimated_gb:.2f} GB', _CYAN)}"
    )

    if skip_score:
        return _make_result(model_id, benchmark_id, status='probe_only',
                            est_gb=est.total_estimated_gb, act_gb=est.activation_gb,
                            feat=est.num_features, stimuli=est.num_stimuli,
                            timebins=est.num_timebins,
                            actual_delta_gb=None, score=None,
                            probe_elapsed=probe_elapsed, score_elapsed=None, note='--skip-score')

    # ── Full benchmark run ───────────────────────────────────────────────
    baseline_rss = proc.memory_info().rss
    _step(f"scoring  (baseline RSS: {_gb(baseline_rss)})")

    # Ticker thread: prints RSS + elapsed every 30s while benchmark runs
    _ticker_stop = threading.Event()
    def _ticker():
        t_start = time.time()
        interval = 30
        while not _ticker_stop.wait(interval):
            elapsed = time.time() - t_start
            rss = proc.memory_info().rss
            print(f"      {_c('…', _DIM)} still scoring  "
                  f"{elapsed/60:.1f} min elapsed  "
                  f"RSS {_gb(rss)}", flush=True)
    ticker_thread = threading.Thread(target=_ticker, daemon=True)
    ticker_thread.start()

    monitor = _PeakMonitor().start()
    t_score = time.time()
    score_val = None
    score_status = 'ok'
    score_note = ''
    try:
        score_val = benchmark(model)
    except MemoryError as e:
        score_status = 'oom'
        score_note = str(e)[:120]
        _substep(_c(f"MemoryError: {score_note}", _RED))
    except Exception as e:
        score_status = 'error'
        score_note = str(e)[:120]
        _substep(_c(f"scoring ERROR: {score_note}", _RED))
    finally:
        _ticker_stop.set()
        ticker_thread.join()

    score_elapsed = time.time() - t_score
    peak_rss = monitor.stop()
    actual_delta_gb = (peak_rss - baseline_rss) / (1024 ** 3)

    # ── Comparison ───────────────────────────────────────────────────────
    colour, verdict, ratio = _compare_label(est.total_estimated_gb, actual_delta_gb)
    _step("comparison")
    _substep(
        f"baseline RSS = {_gb(baseline_rss)}  "
        f"{_c('← model weights, Python, etc. already in RAM before scoring', _DIM)}"
    )
    _substep(
        f"peak RSS     = {_gb(peak_rss)}  "
        f"{_c('← highest point reached during scoring', _DIM)}"
    )
    _substep(
        f"Δ (peak−base)= {_c(f'+{actual_delta_gb:.2f} GB', _CYAN)}  "
        f"{_c('← extra RAM the benchmark itself consumed  ← this is what we compare against', _DIM)}"
    )
    _substep(
        f"estimated    = {_c(f'{est.total_estimated_gb:.2f} GB', _CYAN)}  "
        f"{_c(f'← {est.activation_gb:.3f} GB activations × 6 overhead', _DIM)}"
    )
    _substep(f"verdict      : {_c(verdict, colour)}")
    if score_val is not None:
        _substep(f"score        : {float(score_val):.4f}   elapsed {score_elapsed:.0f}s")

    return _make_result(model_id, benchmark_id,
                        status=score_status,
                        est_gb=est.total_estimated_gb,
                        act_gb=est.activation_gb,
                        feat=est.num_features,
                        stimuli=est.num_stimuli,
                        timebins=est.num_timebins,
                        actual_delta_gb=actual_delta_gb,
                        baseline_rss_gb=baseline_rss / (1024 ** 3),
                        peak_rss_gb=peak_rss / (1024 ** 3),
                        score=float(score_val) if score_val is not None else None,
                        ratio=ratio,
                        probe_elapsed=probe_elapsed,
                        score_elapsed=score_elapsed,
                        note=score_note)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_model(mid):
    return brainscore_vision.load_model(mid)


def _load_benchmark(bid):
    return brainscore_vision.load_benchmark(bid)


def _timed_load(fn, arg, t0, interval=15):
    """Run fn(arg) in a thread, printing elapsed time every `interval` seconds."""
    result = [None]
    exc    = [None]

    def _worker():
        try:
            result[0] = fn(arg)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    while t.is_alive():
        t.join(timeout=interval)
        if t.is_alive():
            print(f"         {_c('…', _DIM)} still loading  {time.time()-t0:.0f}s elapsed",
                  flush=True)

    if exc[0] is not None:
        raise exc[0]
    return result[0]


# ---------------------------------------------------------------------------
# Summary table  (est GB / actual Δ GB per cell)
# ---------------------------------------------------------------------------

_MODEL_W = 36
_CELL_W  = 20


def _trunc(s, n):
    return s if len(s) <= n else s[:n - 1] + '…'


def _cell_text_plain(r):
    """Fixed-width plain text (no ANSI) for padding calculation."""
    est = r.get('est_gb')
    act = r.get('actual_delta_gb')
    if est is None:
        return "  —  skip/err      "
    if act is None:
        return f"{est:5.2f} GB est  n/a  "
    return f"{est:4.1f}/{act:4.1f} GB        "


def _cell_colour(r):
    est = r.get('est_gb')
    act = r.get('actual_delta_gb')
    if est is None:
        return _c("  —  skip/err", _DIM)
    if act is None:
        return f"{_c(f'{est:5.2f} GB', _CYAN)} est"
    _, _, ratio = _compare_label(est, act)
    col = _GREEN if (ratio and ratio >= 0.8) else (_YELLOW if (ratio and ratio >= 0.4) else _RED)
    return f"{_c(f'{est:.1f}', _CYAN)}/{_c(f'{act:.1f}', col)} GB"


def _hline(c_mid, c_left, c_right, c_sep):
    parts = [c_left, c_mid * (_MODEL_W + 2)]
    for _ in BENCHMARKS:
        parts += [c_sep, c_mid * (_CELL_W + 2)]
    parts.append(c_right)
    return ''.join(parts)


def print_summary_table(results_grid):
    top    = _hline('─', '┌', '┐', '┬')
    mid    = _hline('─', '├', '┤', '┼')
    bottom = _hline('─', '└', '┘', '┴')

    print(top)
    hdr = f"│ {_c(_trunc('Model', _MODEL_W), _BOLD):<{_MODEL_W + len(_BOLD) + len(_RESET)}} "
    for bid in BENCHMARKS:
        short = _BM_SHORT.get(bid, bid)
        hdr += f"│ {_c(_trunc(short, _CELL_W), _BOLD):<{_CELL_W + len(_BOLD) + len(_RESET)}} "
    print(hdr + "│")
    print(mid)

    for mid_id in MODELS:
        row = f"│ {_trunc(mid_id, _MODEL_W):<{_MODEL_W}} "
        for bid in BENCHMARKS:
            r = results_grid[mid_id][bid]
            cell = _cell_colour(r)
            padding = _CELL_W - len(_cell_text_plain(r))
            row += f"│ {cell}{' ' * max(0, padding)} "
        print(row + "│")

    print(bottom)
    print(_c("  est GB / actual Δ GB  "
             "(cyan=estimate, green=accurate ≥0.8×, yellow=under 0.4–0.8×, red=under <0.4×)", _DIM))


# ---------------------------------------------------------------------------
# Full text report
# ---------------------------------------------------------------------------

def print_full_report(results_grid):
    print(f"\n{_c('Per-pair results:', _BOLD)}\n")
    for mid_id in MODELS:
        print(f"  {_c(mid_id, _BOLD)}")
        for bid in BENCHMARKS:
            r = results_grid[mid_id][bid]
            short = _BM_SHORT.get(bid, bid)
            est = r.get('est_gb')
            act = r.get('actual_delta_gb')
            score = r.get('score')

            if est is not None and act is not None:
                _, verdict, ratio = _compare_label(est, act)
                col = _GREEN if (ratio and ratio >= 0.8) else (_YELLOW if (ratio and ratio >= 0.4) else _RED)
                score_str = f"   score={score:.4f}" if score is not None else ""
                print(f"    {short:<18}  "
                      f"est {est:.2f} GB  actual Δ {act:.2f} GB  "
                      f"→ {_c(verdict, col)}{score_str}")
            elif est is not None:
                print(f"    {short:<18}  est {est:.2f} GB  no actual  ({r.get('note','')[:50]})")
            else:
                print(f"    {short:<18}  {r['status']}  {r.get('note','')[:60]}")
        print()


# ---------------------------------------------------------------------------
# Overhead recommendation
# ---------------------------------------------------------------------------

def print_overhead_recommendation(results_grid):
    from brainscore_vision.benchmark_helpers.memory import _OVERHEAD_FACTOR

    # Collect pairs where we have both raw activation GB and actual delta GB
    pairs = []
    for mid_id in MODELS:
        for bid in BENCHMARKS:
            r = results_grid[mid_id][bid]
            act_gb  = r.get('act_gb')          # raw activation array GB
            delta_gb = r.get('actual_delta_gb') # actual peak-baseline delta
            if act_gb and act_gb > 0 and delta_gb is not None and delta_gb > 0.01:
                true_factor = delta_gb / act_gb
                pairs.append({
                    'model': r['model_id'],
                    'benchmark': r['benchmark_id'],
                    'act_gb': act_gb,
                    'delta_gb': delta_gb,
                    'true_factor': true_factor,
                })

    n_total = len(MODELS) * len(BENCHMARKS)
    n_scored = len(pairs)

    print(f"\n{'═' * 66}")
    print(f"  {_c('OVERHEAD FACTOR RECOMMENDATION', _BOLD)}")
    print(f"{'═' * 66}\n")

    if n_scored == 0:
        print(f"  {_c('No scored pairs to analyse.', _DIM)}")
        return

    true_factors = sorted(p['true_factor'] for p in pairs)
    current_factor = _OVERHEAD_FACTOR

    # For a given overhead factor F, count pairs where estimate < actual delta
    # (i.e. estimate would have UNDER-predicted, missing a potential OOM)
    def n_underpredicted(factor):
        return sum(1 for p in pairs if p['act_gb'] * factor < p['delta_gb'])

    current_under = n_underpredicted(current_factor)
    n_safe = n_scored - current_under

    print(f"  Scored pairs:  {n_scored}/{n_total}  "
          f"({n_total - n_scored} skipped/errored)\n")
    print(f"  {_c('Current overhead factor = ×{}'.format(current_factor), _BOLD)}")
    print(f"    estimate covered (≥ actual Δ) : "
          f"{_c(str(n_safe), _GREEN)}/{n_scored} pairs")
    print(f"    estimate under-predicted      : "
          f"{_c(str(current_under), _RED)}/{n_scored} pairs  "
          f"{_c('← estimate too low; real usage exceeded prediction', _DIM)}")

    # Show the actual overhead factors observed per pair
    print(f"\n  {_c('Actual overhead factors observed (activation GB → actual Δ GB):', _DIM)}")
    for p in sorted(pairs, key=lambda x: x['true_factor'], reverse=True):
        short_m = p['model'][:28]
        short_b = _BM_SHORT.get(p['benchmark'], p['benchmark'])
        tf = p['true_factor']
        bar = '█' * min(int(tf), 20)
        col = _GREEN if tf <= current_factor else _RED
        factor_str = _c(f'{tf:.1f}×', col)
        print(f"    {short_m:<28}  {short_b:<16}  "
              f"{p['act_gb']:.2f} GB → {p['delta_gb']:.2f} GB  "
              f"= {factor_str}  {_c(bar, col)}")

    # Find the factor that covers each percentile threshold
    print(f"\n  {_c('Factor needed to cover N% of pairs:', _DIM)}")
    for pct in [50, 75, 90, 95, 100]:
        idx = min(int(len(true_factors) * pct / 100), len(true_factors) - 1)
        needed = true_factors[idx]
        rounded = max(current_factor, round(needed + 0.5))  # round up to nearest int
        still_under = n_underpredicted(needed)
        col = _GREEN if needed <= current_factor else _YELLOW if needed <= current_factor * 1.5 else _RED
        print(f"    {pct:>3}% coverage  →  ×{_c(f'{needed:.1f}', col)}  "
              f"(≈ ×{rounded} rounded)  "
              f"→ {still_under}/{n_scored} pairs still under-predicted")

    # Final recommendation: smallest integer factor covering ≥ 90% of pairs
    idx_90 = min(int(len(true_factors) * 0.90), len(true_factors) - 1)
    factor_90 = true_factors[idx_90]
    recommended = max(current_factor, int(factor_90) + (1 if factor_90 % 1 > 0 else 0))
    under_at_rec = n_underpredicted(recommended)

    print(f"\n  {_c('Recommendation', _BOLD)}")
    if recommended == current_factor:
        print(f"    Current factor ×{current_factor} already covers ≥90% of pairs. {_c('No change needed.', _GREEN)}")
    else:
        improvement = current_under - under_at_rec
        print(f"    Increase overhead factor from "
              f"{_c(f'×{current_factor}', _RED)} → {_c(f'×{recommended}', _GREEN)}")
        print(f"    This moves from {_c(str(current_under), _RED)} under-predicted pairs "
              f"to {_c(str(under_at_rec), _GREEN)} "
              f"({_c(f'−{improvement} pairs', _GREEN)} now safely caught)")
        print(f"\n    To apply: set  {_c('_OVERHEAD_FACTOR = ' + str(recommended), _CYAN)}  "
              f"in  brainscore_vision/benchmark_helpers/memory.py")
    print()


# ---------------------------------------------------------------------------
# CSV helpers  (incremental — one row written immediately after each pair)
# ---------------------------------------------------------------------------

_CSV_HEADER = [
    'model', 'benchmark', 'status',
    'est_total_gb', 'act_activation_gb', 'actual_delta_gb',
    'num_features', 'num_stimuli', 'num_timebins',
    'baseline_rss_gb', 'peak_rss_gb',
    'ratio', 'score',
    'probe_elapsed_s', 'score_elapsed_s', 'note',
]


def _csv_row(r):
    def _f(k, fmt='.4f'):
        v = r.get(k)
        return format(v, fmt) if v is not None else ''
    return [
        r['model_id'], r['benchmark_id'], r['status'],
        _f('est_gb'), _f('act_gb'), _f('actual_delta_gb'),
        r.get('feat', ''), r.get('stimuli', ''), r.get('timebins', ''),
        _f('baseline_rss_gb'), _f('peak_rss_gb'),
        _f('ratio'), _f('score'),
        _f('probe_elapsed', '.2f'), _f('score_elapsed', '.2f'),
        r.get('note', ''),
    ]


def init_csv(path):
    """Write header row, return open file handle + csv.writer."""
    f = open(path, 'w', newline='')
    w = csv.writer(f)
    w.writerow(_CSV_HEADER)
    f.flush()
    return f, w


def append_csv_row(writer, file_handle, r):
    writer.writerow(_csv_row(r))
    file_handle.flush()  # write immediately so partial results survive a crash


# ---------------------------------------------------------------------------
# Calibration mode  (alexnet × all benchmarks → fixed_benchmark_cost per bm)
# ---------------------------------------------------------------------------

def run_calibration_pair(model, benchmark, benchmark_id, bm_idx, n_bm):
    """Run one benchmark and return fixed_benchmark_cost = actual_delta - activation_gb."""
    from brainscore_vision.benchmark_helpers.memory import preallocate_memory

    proc = psutil.Process(os.getpid())

    print(f"\n  [{bm_idx}/{n_bm}] {benchmark_id}")
    print(f"  {'─' * 62}")

    # Probe
    _step("probe  (1-stimulus forward pass)")
    try:
        est = preallocate_memory(model, benchmark, raise_if_oom=False)
    except TypeError:
        _substep(_c("skipped — not a NeuralBenchmark (behavioral/non-neural)", _DIM))
        print(f"  {_c(benchmark_id[:55], _DIM)}: N/A (non-neural)", flush=True)
        return dict(benchmark_id=benchmark_id, status='skip',
                    activation_gb=None, actual_delta_gb=None, fixed_cost_gb=None)
    except Exception as e:
        _substep(_c(f"probe ERROR: {str(e)[:80]}", _RED))
        return dict(benchmark_id=benchmark_id, status='error',
                    activation_gb=None, actual_delta_gb=None, fixed_cost_gb=None,
                    note=str(e)[:100])

    if est is None:
        return dict(benchmark_id=benchmark_id, status='skip',
                    activation_gb=None, actual_delta_gb=None, fixed_cost_gb=None)

    _substep(
        f"activation = {est.activation_gb:.3f} GB  "
        f"({est.num_features:,} feat × {est.num_stimuli:,} stim × {est.num_timebins} tbin)"
    )

    # Score
    baseline_rss = proc.memory_info().rss
    _step(f"scoring  (baseline RSS: {_gb(baseline_rss)})")

    _ticker_stop = threading.Event()
    def _ticker():
        t_start = time.time()
        while not _ticker_stop.wait(30):
            elapsed = time.time() - t_start
            rss = proc.memory_info().rss
            print(f"      {_c('…', _DIM)} still scoring  "
                  f"{elapsed/60:.1f} min  RSS {_gb(rss)}", flush=True)
    ticker_thread = threading.Thread(target=_ticker, daemon=True)
    ticker_thread.start()

    monitor = _PeakMonitor().start()
    t_score = time.time()
    score_status = 'ok'
    score_note = ''
    try:
        benchmark(model)
    except MemoryError as e:
        score_status = 'oom'
        score_note = str(e)[:120]
        _substep(_c(f"MemoryError: {score_note}", _RED))
    except Exception as e:
        score_status = 'error'
        score_note = str(e)[:120]
        _substep(_c(f"ERROR: {score_note}", _RED))
    finally:
        _ticker_stop.set()
        ticker_thread.join()

    score_elapsed = time.time() - t_score
    peak_rss = monitor.stop()
    actual_delta_gb = (peak_rss - baseline_rss) / (1024 ** 3)

    fixed_cost_gb = None
    if score_status == 'ok':
        fixed_cost_gb = max(0.0, actual_delta_gb - est.activation_gb)
        _step("result")
        _substep(
            f"actual Δ = {_c(f'{actual_delta_gb:.3f} GB', _CYAN)}  "
            f"elapsed {score_elapsed:.0f}s"
        )
        _substep(
            f"fixed_benchmark_cost = {actual_delta_gb:.3f} − {est.activation_gb:.3f} "
            f"= {_c(f'{fixed_cost_gb:.3f} GB', _GREEN)}"
        )
        # One-liner summary line
        print(
            f"\n  {_c(benchmark_id[:55], _BOLD)}: "
            f"fixed_cost = {_c(f'{fixed_cost_gb:.2f} GB', _GREEN)}",
            flush=True,
        )
    else:
        _substep(f"actual Δ = {_gb(peak_rss - baseline_rss)}  (run failed — no fixed_cost)")

    return dict(
        benchmark_id=benchmark_id,
        status=score_status,
        activation_gb=est.activation_gb,
        actual_delta_gb=actual_delta_gb,
        fixed_cost_gb=fixed_cost_gb,
        score_elapsed=score_elapsed,
        note=score_note,
    )


def print_calibration_table(results):
    neural = [r for r in results if r.get('fixed_cost_gb') is not None]
    skipped = sum(1 for r in results if r['status'] == 'skip')
    errors  = sum(1 for r in results if r['status'] in ('error', 'oom'))

    print(f"\n\n{'═' * 72}")
    print(f"  {_c('BENCHMARK FIXED COSTS  (model-independent overhead)', _BOLD)}")
    print(f"  Calibrated with alexnet on {len(neural)} neural benchmarks  "
          f"|  {skipped} non-neural skipped  |  {errors} errors")
    print(f"{'═' * 72}\n")

    if not neural:
        print(f"  {_c('No neural benchmarks scored.', _DIM)}")
        return

    neural_sorted = sorted(neural, key=lambda r: r['fixed_cost_gb'], reverse=True)
    col_w = min(max(len(r['benchmark_id']) for r in neural_sorted), 56)

    print(f"  {'Benchmark':<{col_w}}  {'Fixed cost':>12}  {'Act. Δ':>9}  {'Activation':>10}")
    print(f"  {'─' * col_w}  {'─' * 12}  {'─' * 9}  {'─' * 10}")

    for r in neural_sorted:
        bid  = r['benchmark_id'][:col_w]
        fc   = r['fixed_cost_gb']
        act  = r['actual_delta_gb']
        actl = r['activation_gb']
        col  = _GREEN if fc < 5 else (_YELLOW if fc < 15 else _RED)
        print(
            f"  {bid:<{col_w}}  "
            f"{_c(f'{fc:>8.2f} GB', col)}  "
            f"{act:>7.2f} GB  "
            f"{actl:>8.3f} GB"
        )

    print()
    print(f"  {_c('Formula:', _BOLD)} total_needed = activation_gb + fixed_benchmark_cost")
    print(f"  {_c('Usage:', _DIM)}   preallocate_memory(model, bm, "
          f"fixed_benchmark_cost_gb=<value>)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_calibrate(args):
    """--calibrate mode: load alexnet, run every benchmark, output fixed_benchmark_cost."""
    from brainscore_vision.benchmark_helpers.memory import save_calibration, _DEFAULT_CALIBRATION_PATH
    cal_path = getattr(args, 'calibration_json', None) or _DEFAULT_CALIBRATION_PATH

    n_bm = len(ALL_BENCHMARKS)
    print(f"\n{'═' * 72}")
    print(f"  {_c('CALIBRATION MODE', _BOLD)}  —  alexnet × {n_bm} benchmarks")
    print(f"  Goal: measure fixed_benchmark_cost = actual_Δ − activation_gb per benchmark")
    print(f"  Calibration table will be saved → {_c(cal_path, _CYAN)}")
    print(f"{'═' * 72}\n")

    # Load alexnet
    _step("loading alexnet...", indent=2)
    t0 = time.time()
    try:
        model = _timed_load(_load_model, 'alexnet', t0)
        print(f"       {_c('OK', _GREEN)} ({time.time() - t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"       {_c('FAILED', _RED)}: {e}")
        return

    # Open CSV
    csv_file, csv_writer = None, None
    if args.csv:
        csv_file = open(args.csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['benchmark', 'status', 'activation_gb',
                             'actual_delta_gb', 'fixed_cost_gb', 'score_elapsed_s', 'note'])
        csv_file.flush()
        print(f"\n  {_c('CSV →', _CYAN)} {args.csv}\n")

    results = []
    costs = {}  # live-updated so JSON is always current
    try:
        for i, bid in enumerate(ALL_BENCHMARKS, 1):
            # Load benchmark with ticker
            t0 = time.time()
            try:
                bm = _timed_load(_load_benchmark, bid, t0)
            except Exception as e:
                print(f"\n  [{i}/{n_bm}] {bid}")
                _substep(_c(f"load FAILED: {str(e)[:80]}", _RED))
                r = dict(benchmark_id=bid, status='error', activation_gb=None,
                         actual_delta_gb=None, fixed_cost_gb=None, note=str(e)[:100])
                results.append(r)
                if csv_writer:
                    csv_writer.writerow([bid, 'error', '', '', '', '', r.get('note', '')])
                    csv_file.flush()
                continue

            r = run_calibration_pair(model, bm, bid, i, n_bm)
            results.append(r)

            if csv_writer:
                csv_writer.writerow([
                    bid, r['status'],
                    f"{r['activation_gb']:.4f}"  if r['activation_gb']  is not None else '',
                    f"{r['actual_delta_gb']:.4f}" if r['actual_delta_gb'] is not None else '',
                    f"{r['fixed_cost_gb']:.4f}"  if r['fixed_cost_gb']   is not None else '',
                    f"{r.get('score_elapsed', ''):.1f}" if r.get('score_elapsed') else '',
                    r.get('note', ''),
                ])
                csv_file.flush()

            # Incrementally save JSON after every benchmark that yielded a cost
            if r.get('fixed_cost_gb') is not None:
                costs[bid] = r['fixed_cost_gb']
                save_calibration(costs, cal_path)
                print(f"  {_c('↳ JSON updated', _DIM)} ({len(costs)} benchmarks so far)",
                      flush=True)

    finally:
        if csv_file:
            csv_file.close()
            print(f"\n{_c('CSV finalised →', _CYAN)} {args.csv}")

    save_calibration(costs, cal_path)
    print(f"\n{_c('Calibration saved →', _CYAN)} {cal_path}  ({len(costs)} benchmarks)\n")

    print_calibration_table(results)


def main():
    parser = argparse.ArgumentParser(
        description="Memory profile suite — 5×5 estimate vs actual, or benchmark calibration.")
    parser.add_argument('--csv', metavar='PATH', default=None,
                        help='write results to CSV')
    parser.add_argument('--skip-score', action='store_true',
                        help='probe only — do not run actual scoring')
    parser.add_argument('--calibrate', action='store_true',
                        help='run alexnet on ALL benchmarks and output fixed_benchmark_cost per benchmark')
    parser.add_argument('--calibration-json', metavar='PATH', default=None,
                        help='path to save/load calibration JSON '
                             '(default: ~/.brainscore/benchmark_costs.json)')
    args = parser.parse_args()

    if args.calibrate:
        _run_calibrate(args)
        return

    n_bm = len(BENCHMARKS)
    n_m  = len(MODELS)
    mode = "probe only" if args.skip_score else "probe + score"
    total_pairs = n_m * n_bm

    print(f"\n{'═' * 66}")
    print(f"  {_c('MEM PROFILE SUITE', _BOLD)}  —  "
          f"{n_m} models × {n_bm} benchmarks = {total_pairs} pairs  [{mode}]")
    print(f"{'═' * 66}")

    # ── Load all benchmarks first ────────────────────────────────────────
    print(f"\n{_c(f'Loading {n_bm} benchmarks  (may download from S3)', _CYAN)}\n")
    benchmarks = {}
    for i, bid in enumerate(BENCHMARKS, 1):
        print(f"  [{i}/{n_bm}] {bid}")
        t0 = time.time()
        try:
            benchmarks[bid] = _timed_load(_load_benchmark, bid, t0)
            print(f"         {_c('OK', _GREEN)} ({time.time() - t0:.1f}s)")
        except Exception as e:
            benchmarks[bid] = None
            print(f"         {_c('FAILED', _RED)}: {str(e)[:80]}")
        print()

    # ── Open CSV for incremental writing ────────────────────────────────
    csv_file, csv_writer = None, None
    if args.csv:
        csv_file, csv_writer = init_csv(args.csv)
        print(f"  {_c('CSV opened →', _CYAN)} {args.csv}  (rows written after each pair)\n")

    # ── For each model, run all benchmarks ───────────────────────────────
    results_grid = {mid_id: {} for mid_id in MODELS}
    pair_num = 0

    try:
        for m_idx, mid_id in enumerate(MODELS, 1):
            print(f"\n{'═' * 66}")
            print(f"  {_c(f'Model {m_idx}/{n_m}: {mid_id}', _CYAN)}")
            print(f"{'═' * 66}\n")

            _step("loading model...", indent=2)
            t0 = time.time()
            try:
                model = _timed_load(_load_model, mid_id, t0)
                print(f"       {_c('OK', _GREEN)} ({time.time() - t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"       {_c('FAILED', _RED)}: {str(e)[:80]}")
                for bid in BENCHMARKS:
                    pair_num += 1
                    r = _make_result(
                        mid_id, bid, status='error', est_gb=None, act_gb=None,
                        actual_delta_gb=None, score=None,
                        probe_elapsed=0.0, score_elapsed=None,
                        note=f"model load failed: {str(e)[:60]}")
                    results_grid[mid_id][bid] = r
                    if csv_writer:
                        append_csv_row(csv_writer, csv_file, r)
                continue

            for bid in BENCHMARKS:
                pair_num += 1
                short = _BM_SHORT.get(bid, bid)
                print(f"\n  {_c(f'pair {pair_num}/{total_pairs}', _DIM)}  "
                      f"{_c(_trunc(mid_id, 28), _BOLD)} × {_c(short, _BOLD)}")
                print(f"  {'─' * 54}")

                bm = benchmarks.get(bid)
                if bm is None:
                    _step(_c("benchmark failed to load — skipping", _RED), indent=4)
                    r = _make_result(
                        mid_id, bid, status='error', est_gb=None, act_gb=None,
                        actual_delta_gb=None, score=None,
                        probe_elapsed=0.0, score_elapsed=None,
                        note='benchmark failed to load')
                    results_grid[mid_id][bid] = r
                    if csv_writer:
                        append_csv_row(csv_writer, csv_file, r)
                    continue

                r = run_pair(model, mid_id, bm, bid, skip_score=args.skip_score)
                results_grid[mid_id][bid] = r

                # Write CSV row immediately
                if csv_writer:
                    append_csv_row(csv_writer, csv_file, r)
                    print(f"  {_c('↳ CSV row written', _DIM)}", flush=True)

                # One-line pair summary
                est = r.get('est_gb')
                act = r.get('actual_delta_gb')
                if est is not None and act is not None:
                    _, verdict, ratio = _compare_label(est, act)
                    col = (_GREEN if (ratio and ratio >= 0.8)
                           else _YELLOW if (ratio and ratio >= 0.4) else _RED)
                    print(f"\n  {_c('RESULT', _BOLD)}: "
                          f"est {est:.2f} GB  actual Δ {act:.2f} GB  → {_c(verdict, col)}")
                elif est is not None:
                    print(f"\n  {_c('RESULT', _BOLD)}: est {est:.2f} GB  (no actual)")

    finally:
        if csv_file:
            csv_file.close()
            print(f"\n{_c('CSV finalised →', _CYAN)} {args.csv}")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n\n{'═' * 66}")
    print(f"  {_c('FINAL SUMMARY', _BOLD)}")
    print(f"{'═' * 66}\n")
    print_summary_table(results_grid)
    print_full_report(results_grid)
    print_overhead_recommendation(results_grid)


if __name__ == '__main__':
    main()
