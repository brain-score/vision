"""
Pre-flight Memory Check
=======================
The main entry point for checking whether a model will OOM on a benchmark
before committing to a full (potentially multi-hour) scoring run.

HOW IT WORKS
------------
1. Loads the calibration table from ~/.brainscore/benchmark_costs.json
   (produced by:  python scripts/mem_profile_suite.py --calibrate)

2. Runs a 1-stimulus forward pass through the model (the "probe") to measure
   the model's actual feature count for this benchmark's region/layer.

3. Estimates total RAM needed:
     total = activation_gb + fixed_benchmark_cost_gb   (if benchmark is calibrated)
     total = activation_gb × 6                          (fallback if not calibrated)

   where:
     activation_gb          = stimuli × features × timebins × 4 bytes
     fixed_benchmark_cost   = benchmark's model-independent overhead
                              (regression matrices, xarray, CV buffers)
                              — constant regardless of which model you run

4. Compares the estimate against available RAM and reports OK or OOM LIKELY.

Optionally (--score) runs the full benchmark and compares the estimate to
the actual peak RSS delta, so you can validate the calibration on this machine.

IMPORTANT: Calibrate on the same machine you score on.  The fixed_benchmark_cost
is environment-specific (Linux EC2 numbers will differ from macOS).

Usage
-----
    python scripts/preflight_check.py <model_id> <benchmark_id> [--score]

Examples
--------
    # Fast probe — just check if it will OOM (recommended before any scoring run)
    python scripts/preflight_check.py resnet50_tutorial MajajHong2015.IT-pls

    # Full roundtrip — probe then score and compare estimate to actual peak RSS
    python scripts/preflight_check.py resnet50_tutorial MajajHong2015.IT-pls --score
"""

import os
import sys
import time
import argparse
import threading

# ---------------------------------------------------------------------------
# Resolve local repos
# ---------------------------------------------------------------------------
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_vision_root = os.path.dirname(_script_dir)
_core_root   = os.path.join(os.path.dirname(_vision_root), 'core')
for _p in [_vision_root, _core_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging
logging.basicConfig(level=logging.WARNING)

import psutil

_RESET  = '\033[0m'
_BOLD   = '\033[1m'
_GREEN  = '\033[32m'
_YELLOW = '\033[33m'
_RED    = '\033[31m'
_CYAN   = '\033[36m'
_DIM    = '\033[2m'


def _c(text, colour):
    return f"{colour}{text}{_RESET}"


def _gb(n_bytes):
    return f"{n_bytes / (1024 ** 3):.2f} GB"


def _divider(char='─', width=66):
    print(char * width)


# ---------------------------------------------------------------------------
# Peak RSS monitor
# ---------------------------------------------------------------------------
class _PeakMonitor:
    def __init__(self, interval=0.5):
        self._proc = psutil.Process(os.getpid())
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
            self._stop.wait(0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Pre-flight memory check integration test.')
    parser.add_argument('model_id')
    parser.add_argument('benchmark_id')
    parser.add_argument('--score', action='store_true',
                        help='also run the full benchmark and compare estimate to actual RSS')
    args = parser.parse_args()

    from brainscore_vision import load_model, load_benchmark
    from brainscore_vision.benchmark_helpers.memory import (
        preallocate_memory, load_calibration, _DEFAULT_CALIBRATION_PATH,
    )

    print()
    _divider('═')
    print(f"  {_c('PRE-FLIGHT CHECK', _BOLD)}")
    print(f"  model     : {_c(args.model_id, _CYAN)}")
    print(f"  benchmark : {_c(args.benchmark_id, _CYAN)}")
    _divider('═')

    # ── Calibration table status ─────────────────────────────────────────
    print()
    cal = load_calibration()
    if cal:
        fixed = cal.get(args.benchmark_id)
        if fixed is not None:
            print(f"  {_c('Calibration table', _BOLD)}: {_DEFAULT_CALIBRATION_PATH}")
            print(f"  {_c('✓', _GREEN)} '{args.benchmark_id}' found  →  "
                  f"fixed_benchmark_cost = {_c(f'{fixed:.4f} GB', _GREEN)}")
            print(f"  Formula: total = activation_gb + {fixed:.4f} GB")
        else:
            print(f"  {_c('Calibration table', _BOLD)}: {_DEFAULT_CALIBRATION_PATH}  "
                  f"({len(cal)} entries)")
            print(f"  {_c('⚠', _YELLOW)} '{args.benchmark_id}' not in table  →  "
                  f"will fall back to ×6 overhead")
    else:
        print(f"  {_c('⚠', _YELLOW)} No calibration table found at {_DEFAULT_CALIBRATION_PATH}")
        print(f"  Will fall back to ×6 overhead multiplier.")

    # ── Load model + benchmark ───────────────────────────────────────────
    print()
    _divider()

    print(f"\n  Loading model '{args.model_id}'...", end='', flush=True)
    t0 = time.time()
    model = load_model(args.model_id)
    print(f"  {_c('OK', _GREEN)} ({time.time()-t0:.1f}s)")

    print(f"  Loading benchmark '{args.benchmark_id}'...", end='', flush=True)
    t0 = time.time()
    benchmark = load_benchmark(args.benchmark_id)
    print(f"  {_c('OK', _GREEN)} ({time.time()-t0:.1f}s)")

    # ── Pre-flight probe ─────────────────────────────────────────────────
    print()
    _divider()
    print(f"\n  {_c('PRE-FLIGHT PROBE', _BOLD)}  (1-stimulus forward pass)\n")

    t0 = time.time()
    try:
        est = preallocate_memory(model, benchmark, raise_if_oom=False)
    except TypeError as e:
        print(f"  {_c('SKIPPED', _YELLOW)}: {e}")
        return

    probe_elapsed = time.time() - t0

    if est is None:
        print(f"  {_c('SKIPPED', _YELLOW)} (BRAINSCORE_SKIP_MEMORY_CHECK set)")
        return

    print(f"  {'Stimuli':<22}: {est.num_stimuli:,}")
    print(f"  {'Features (neuroid)':<22}: {est.num_features:,}")
    print(f"  {'Timebins':<22}: {est.num_timebins}")
    print(f"  {'Activation array':<22}: {est.activation_gb:.4f} GB  "
          f"{_c(f'({est.num_stimuli} × {est.num_features:,} × {est.num_timebins} × 4B)', _DIM)}")
    print()

    if est.fixed_benchmark_cost_gb is not None:
        print(f"  {_c('Formula', _BOLD)}: {_c('CALIBRATED', _GREEN)}")
        print(f"  {'Activation':<22}: {est.activation_gb:.4f} GB")
        print(f"  {'Fixed benchmark cost':<22}: {est.fixed_benchmark_cost_gb:.4f} GB  "
              f"{_c('← model-independent overhead from calibration table', _DIM)}")
        print(f"  {'Total estimated':<22}: {_c(f'{est.total_estimated_gb:.4f} GB', _CYAN)}  "
              f"{_c(f'({est.activation_gb:.4f} + {est.fixed_benchmark_cost_gb:.4f})', _DIM)}")
    else:
        print(f"  {_c('Formula', _BOLD)}: {_c('FALLBACK (×6)', _YELLOW)}  "
              f"{_c('← benchmark not in calibration table', _DIM)}")
        print(f"  {'Activation':<22}: {est.activation_gb:.4f} GB")
        print(f"  {'Total estimated':<22}: {_c(f'{est.total_estimated_gb:.4f} GB', _CYAN)}  "
              f"{_c(f'({est.activation_gb:.4f} × 6)', _DIM)}")

    print()
    avail_col = _RED if est.will_oom else _GREEN
    verdict = _c('OOM LIKELY', _RED) if est.will_oom else _c('OK', _GREEN)
    print(f"  {'Available RAM':<22}: {_c(f'{est.available_gb:.2f} GB', avail_col)}")
    print(f"  {'Verdict':<22}: {verdict}")
    print(f"  {'Probe elapsed':<22}: {probe_elapsed:.1f}s")

    if not args.score:
        print()
        _divider()
        print(f"\n  {_c('Tip:', _DIM)} run with --score to also execute the full benchmark")
        print(f"  and compare the estimate against actual peak RSS.\n")
        return

    # ── Full benchmark run ───────────────────────────────────────────────
    print()
    _divider()
    print(f"\n  {_c('FULL BENCHMARK RUN', _BOLD)}\n")

    proc = psutil.Process(os.getpid())
    baseline_rss = proc.memory_info().rss
    print(f"  Baseline RSS: {_gb(baseline_rss)}  "
          f"{_c('← everything already in RAM (model weights, Python, etc.)', _DIM)}")
    print(f"  Scoring...  (this may take a while)\n")

    # Ticker thread
    _ticker_stop = threading.Event()
    def _ticker():
        t_start = time.time()
        while not _ticker_stop.wait(30):
            elapsed = time.time() - t_start
            rss = proc.memory_info().rss
            print(f"    {_c('…', _DIM)} still scoring  {elapsed/60:.1f} min  RSS {_gb(rss)}",
                  flush=True)
    ticker_thread = threading.Thread(target=_ticker, daemon=True)
    ticker_thread.start()

    monitor = _PeakMonitor().start()
    t_score = time.time()
    score_val = None
    try:
        score_val = benchmark(model)
    except MemoryError as e:
        print(f"\n  {_c('MemoryError', _RED)}: {e}")
    except Exception as e:
        print(f"\n  {_c('ERROR', _RED)}: {e}")
    finally:
        _ticker_stop.set()
        ticker_thread.join()

    score_elapsed = time.time() - t_score
    peak_rss  = monitor.stop()
    final_rss = proc.memory_info().rss
    actual_delta_gb = (peak_rss - baseline_rss) / (1024 ** 3)

    # ── Comparison ───────────────────────────────────────────────────────
    print()
    _divider()
    print(f"\n  {_c('RESULT', _BOLD)}\n")

    print(f"  {'Baseline RSS':<24}: {_gb(baseline_rss)}")
    print(f"  {'Peak RSS':<24}: {_gb(peak_rss)}")
    print(f"  {'Δ (peak − baseline)':<24}: {_c(f'+{actual_delta_gb:.4f} GB', _CYAN)}  "
          f"{_c('← actual RAM the benchmark consumed', _DIM)}")
    print(f"  {'Estimated total':<24}: {_c(f'{est.total_estimated_gb:.4f} GB', _CYAN)}")
    print()

    if actual_delta_gb > 0.01:
        error_gb  = est.total_estimated_gb - actual_delta_gb
        error_pct = (error_gb / actual_delta_gb) * 100
        if error_gb >= 0:
            accuracy = _c(f'OVER by {error_gb:.2f} GB ({error_pct:.1f}%)  ← conservative, safe', _GREEN)
        elif abs(error_pct) <= 15:
            accuracy = _c(f'UNDER by {abs(error_gb):.2f} GB ({abs(error_pct):.1f}%)  ← within 15%, acceptable', _YELLOW)
        else:
            accuracy = _c(f'UNDER by {abs(error_gb):.2f} GB ({abs(error_pct):.1f}%)  ← significant miss', _RED)
        print(f"  {'Accuracy':<24}: {accuracy}")

        formula = 'calibrated' if est.fixed_benchmark_cost_gb is not None else '×6 fallback'
        print(f"  {'Formula used':<24}: {formula}")

    if score_val is not None:
        print(f"  {'Score':<24}: {float(score_val):.4f}")
    print(f"  {'Elapsed':<24}: {score_elapsed:.0f}s")
    print()
    _divider('═')
    print()


if __name__ == '__main__':
    main()
