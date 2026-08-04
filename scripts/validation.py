"""
Pre-flight Estimator Validation
================================
Runs a 3-model × 4-benchmark grid to validate the pre-flight memory estimator
across all formula types: PLS, Ridge (calibrated), RidgeCV (calibrated), and RDM.

One benchmark is selected per formula class so that every code path in
preallocate_memory is exercised:

  FreemanZiemba2013.V1-pls   → PLS  (activation × 7 + fixed_cost, warning printed)
  Papale2025.IT-ridge        → Ridge calibrated  (activation + calibrated cost)
  Gifford2022.IT-ridgecv     → RidgeCV calibrated  (activation + calibrated cost)
  Allen2022_fmri.IT-rdm      → RDM  (activation + 2×n_stimuli²×4B, model-independent)

For each (model, benchmark) pair it:
  1. Runs the pre-flight probe  → estimates total GB via the appropriate formula
  2. Runs the full benchmark    → measures actual peak RSS delta
  3. Compares estimate to actual and reports over/under by how much

Results are written to `validation_results.jsonl` after every pair so a crash
does not lose completed work.  Re-running will overwrite the file.

Usage
-----
    python scripts/validation.py

    # Skip the actual benchmark runs (probe only — fast)
    python scripts/validation.py --probe-only

    # Write results to a custom path
    python scripts/validation.py --output /tmp/val.jsonl
"""

import argparse
import json
import logging
import os
import sys
import threading
import time

import psutil

# ---------------------------------------------------------------------------
# Resolve local repos so the script works without installation
# ---------------------------------------------------------------------------
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_vision_root = os.path.dirname(_script_dir)
_core_root   = os.path.join(os.path.dirname(_vision_root), 'core')
for _p in [_vision_root, _core_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
MODELS = [
    'alexnet',
    'resnet50_tutorial',
    'vit_large_patch14_clip_224:openai_ft_in1k',
]

BENCHMARKS = [
    'FreemanZiemba2013.V1-pls',    # PLS         — activation × 7 + fixed_cost (approximate, warning)
    'Papale2025.IT-ridge',          # Ridge        — activation + calibrated cost
    'Gifford2022.IT-ridgecv',       # RidgeCV      — activation + calibrated cost
    'Allen2022_fmri.IT-rdm',        # RDM          — activation + 2×n_stimuli²×4B (model-independent)
]

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
_RESET  = '\033[0m'
_BOLD   = '\033[1m'
_DIM    = '\033[2m'
_GREEN  = '\033[32m'
_YELLOW = '\033[33m'
_RED    = '\033[31m'
_CYAN   = '\033[36m'

def _c(text, colour): return f"{colour}{text}{_RESET}"
def _gb(n_bytes):     return f"{n_bytes / (1024 ** 3):.3f} GB"


# ---------------------------------------------------------------------------
# Peak RSS monitor
# ---------------------------------------------------------------------------
class _PeakMonitor:
    def __init__(self, interval=0.5):
        self._proc  = psutil.Process(os.getpid())
        self._peak  = self._proc.memory_info().rss
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self) -> int:
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
# Formatting helpers
# ---------------------------------------------------------------------------
def _divider(char='─', width=70):
    print(char * width)


def _accuracy_label(estimated_gb: float, actual_gb: float):
    """Return a coloured accuracy string."""
    if actual_gb <= 0.01:
        return _c('actual delta too small to measure', _DIM)
    error_gb  = estimated_gb - actual_gb
    error_pct = (error_gb / actual_gb) * 100
    if error_gb >= 0:
        return _c(f'OVER  by {error_gb:.2f} GB ({error_pct:.1f}%)  ← conservative, safe', _GREEN)
    elif abs(error_pct) <= 15:
        return _c(f'UNDER by {abs(error_gb):.2f} GB ({abs(error_pct):.1f}%)  ← within 15%, acceptable', _YELLOW)
    else:
        return _c(f'UNDER by {abs(error_gb):.2f} GB ({abs(error_pct):.1f}%)  ← significant miss', _RED)


def _write_result(path: str, record: dict):
    """Append one JSON record to the output file (crash-safe)."""
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')


# ---------------------------------------------------------------------------
# Single-pair validation
# ---------------------------------------------------------------------------
def run_pair(model_id: str, benchmark_id: str, output_path: str, probe_only: bool) -> dict:
    from brainscore_vision import load_model, load_benchmark
    from brainscore_vision.benchmark_helpers.memory import preallocate_memory

    record = {
        'model': model_id,
        'benchmark': benchmark_id,
        'status': 'pending',
    }

    proc = psutil.Process(os.getpid())

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\n  Loading model     {_c(model_id, _CYAN)} ...", end='', flush=True)
    t0 = time.time()
    try:
        model = load_model(model_id)
    except Exception as e:
        print(f"  {_c('FAILED', _RED)}: {e}")
        record.update({'status': 'error', 'error': f'load_model: {e}'})
        _write_result(output_path, record)
        return record
    print(f"  {_c('OK', _GREEN)} ({time.time()-t0:.1f}s)")

    # ── Load benchmark ───────────────────────────────────────────────────
    print(f"  Loading benchmark {_c(benchmark_id, _CYAN)} ...", end='', flush=True)
    t0 = time.time()
    try:
        benchmark = load_benchmark(benchmark_id)
    except Exception as e:
        print(f"  {_c('FAILED', _RED)}: {e}")
        record.update({'status': 'error', 'error': f'load_benchmark: {e}'})
        _write_result(output_path, record)
        return record
    print(f"  {_c('OK', _GREEN)} ({time.time()-t0:.1f}s)")

    # ── Pre-flight probe ─────────────────────────────────────────────────
    print(f"\n  {_c('PRE-FLIGHT PROBE', _BOLD)}")
    t0 = time.time()
    try:
        est = preallocate_memory(model, benchmark, raise_if_oom=False)
    except TypeError as e:
        print(f"  {_c('SKIPPED', _YELLOW)}: {e}")
        record.update({'status': 'skipped', 'skip_reason': str(e)})
        _write_result(output_path, record)
        return record
    probe_elapsed = time.time() - t0

    if est is None:
        print(f"  {_c('SKIPPED', _YELLOW)} (BRAINSCORE_SKIP_MEMORY_CHECK set)")
        record.update({'status': 'skipped', 'skip_reason': 'BRAINSCORE_SKIP_MEMORY_CHECK'})
        _write_result(output_path, record)
        return record

    formula = 'calibrated' if est.fixed_benchmark_cost_gb is not None else f'x{6}_fallback'
    print(f"  {'Stimuli':<24}: {est.num_stimuli:,}")
    print(f"  {'Features':<24}: {est.num_features:,}")
    print(f"  {'Timebins':<24}: {est.num_timebins}")
    print(f"  {'Activation':<24}: {est.activation_gb:.4f} GB  "
          f"{_c(f'({est.num_stimuli:,} × {est.num_features:,} × {est.num_timebins} × 4B)', _DIM)}")
    if est.fixed_benchmark_cost_gb is not None:
        print(f"  {'Fixed benchmark cost':<24}: {est.fixed_benchmark_cost_gb:.4f} GB  "
              f"{_c('← from calibration table', _DIM)}")
        print(f"  {'Estimated total':<24}: {_c(f'{est.total_estimated_gb:.4f} GB', _CYAN)}  "
              f"{_c(f'({est.activation_gb:.4f} + {est.fixed_benchmark_cost_gb:.4f})', _DIM)}")
    else:
        print(f"  {'Estimated total':<24}: {_c(f'{est.total_estimated_gb:.4f} GB', _CYAN)}  "
              f"{_c(f'({est.activation_gb:.4f} × 6 fallback)', _DIM)}")
    print(f"  {'Available RAM':<24}: {est.available_gb:.2f} GB")
    print(f"  {'OOM predicted':<24}: {_c('YES', _RED) if est.will_oom else _c('NO', _GREEN)}")
    print(f"  {'Probe elapsed':<24}: {probe_elapsed:.1f}s")

    record.update({
        'num_stimuli':           est.num_stimuli,
        'num_features':          est.num_features,
        'num_timebins':          est.num_timebins,
        'activation_gb':         round(est.activation_gb, 6),
        'fixed_benchmark_cost_gb': round(est.fixed_benchmark_cost_gb, 6) if est.fixed_benchmark_cost_gb is not None else None,
        'estimated_total_gb':    round(est.total_estimated_gb, 6),
        'available_gb':          round(est.available_gb, 2),
        'oom_predicted':         est.will_oom,
        'formula':               formula,
        'probe_elapsed_s':       round(probe_elapsed, 1),
    })

    if probe_only:
        record['status'] = 'probe_only'
        _write_result(output_path, record)
        return record

    # ── Full benchmark run ───────────────────────────────────────────────
    print(f"\n  {_c('FULL BENCHMARK RUN', _BOLD)}")
    baseline_rss = proc.memory_info().rss
    print(f"  Baseline RSS: {_gb(baseline_rss)}  "
          f"{_c('← model weights + Python already in RAM', _DIM)}")
    print(f"  Scoring...  (this may take a while)", flush=True)

    # Ticker thread — prints a heartbeat every 60s so we know it's alive
    _ticker_stop = threading.Event()
    def _ticker():
        t_start = time.time()
        while not _ticker_stop.wait(60):
            elapsed = time.time() - t_start
            rss = proc.memory_info().rss
            print(f"    {_c('…', _DIM)} still scoring  {elapsed/60:.1f} min  RSS {_gb(rss)}", flush=True)
    ticker = threading.Thread(target=_ticker, daemon=True)
    ticker.start()

    monitor = _PeakMonitor().start()
    t_score = time.time()
    score_val = None
    score_error = None
    try:
        score_val = benchmark(model)
    except MemoryError as e:
        score_error = f'MemoryError: {e}'
        print(f"\n  {_c('MemoryError', _RED)}: {e}")
    except Exception as e:
        score_error = f'{type(e).__name__}: {e}'
        print(f"\n  {_c('ERROR', _RED)} ({type(e).__name__}): {e}")
    finally:
        _ticker_stop.set()
        ticker.join()

    score_elapsed  = time.time() - t_score
    peak_rss       = monitor.stop()
    actual_delta_gb = (peak_rss - baseline_rss) / (1024 ** 3)

    # ── Comparison ───────────────────────────────────────────────────────
    print(f"\n  {_c('COMPARISON', _BOLD)}")
    print(f"  {'Baseline RSS':<24}: {_gb(baseline_rss)}")
    print(f"  {'Peak RSS':<24}: {_gb(peak_rss)}")
    print(f"  {'Actual delta':<24}: {_c(f'+{actual_delta_gb:.4f} GB', _CYAN)}  "
          f"{_c('← RAM the benchmark consumed', _DIM)}")
    print(f"  {'Estimated total':<24}: {_c(f'{est.total_estimated_gb:.4f} GB', _CYAN)}")
    print(f"  {'Accuracy':<24}: {_accuracy_label(est.total_estimated_gb, actual_delta_gb)}")
    print(f"  {'Score elapsed':<24}: {score_elapsed:.0f}s")
    if score_val is not None:
        print(f"  {'Score':<24}: {float(score_val):.4f}")

    record.update({
        'baseline_rss_gb':   round(baseline_rss / (1024 ** 3), 4),
        'peak_rss_gb':       round(peak_rss / (1024 ** 3), 4),
        'actual_delta_gb':   round(actual_delta_gb, 4),
        'error_gb':          round(est.total_estimated_gb - actual_delta_gb, 4),
        'error_pct':         round((est.total_estimated_gb - actual_delta_gb) / actual_delta_gb * 100, 1)
                             if actual_delta_gb > 0.01 else None,
        'score_elapsed_s':   round(score_elapsed, 0),
        'score':             float(score_val) if score_val is not None else None,
        'score_error':       score_error,
        'status':            'error' if score_error else 'ok',
    })
    _write_result(output_path, record)
    return record


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(results: list[dict]):
    print()
    _divider('═')
    print(f"  {_c('VALIDATION SUMMARY', _BOLD)}  ({len(results)} pairs)\n")

    header = f"  {'Model':<48}  {'Benchmark':<30}  {'Est GB':>8}  {'Act GB':>8}  {'Err GB':>8}  {'Err %':>7}  Status"
    print(header)
    _divider()

    for r in results:
        model = r['model'][-46:]   # truncate long model names
        bm    = r['benchmark']
        est   = r.get('estimated_total_gb')
        act   = r.get('actual_delta_gb')
        err   = r.get('error_gb')
        pct   = r.get('error_pct')
        status = r.get('status', '?')

        if status == 'ok':
            if err is not None and err >= 0:
                status_str = _c('OVER', _GREEN)
            elif pct is not None and abs(pct) <= 15:
                status_str = _c('~OK', _YELLOW)
            else:
                status_str = _c('MISS', _RED)
        elif status in ('skipped', 'skipped_oom', 'probe_only'):
            status_str = _c(status.upper(), _YELLOW)
        else:
            status_str = _c(status.upper(), _RED)

        est_s = f"{est:.3f}" if est is not None else '—'
        act_s = f"{act:.3f}" if act is not None else '—'
        err_s = f"{err:+.3f}" if err is not None else '—'
        pct_s = f"{pct:+.1f}%" if pct is not None else '—'

        print(f"  {model:<48}  {bm:<30}  {est_s:>8}  {act_s:>8}  {err_s:>8}  {pct_s:>7}  {status_str}")

    _divider('═')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Pre-flight estimator validation suite.')
    parser.add_argument('--probe-only', action='store_true',
                        help='Only run the pre-flight probe; skip full benchmark scoring.')
    parser.add_argument('--output', default=os.path.join(_script_dir, 'validation_results.jsonl'),
                        help='Path to write per-pair JSONL results (default: scripts/validation_results.jsonl)')
    args = parser.parse_args()

    # Truncate output file at the start of a fresh run
    open(args.output, 'w').close()
    print(f"\n{_c('Results will be written incrementally to:', _DIM)} {args.output}\n")

    n_pairs = len(MODELS) * len(BENCHMARKS)
    pair_idx = 0
    results = []

    for model_id in MODELS:
        for benchmark_id in BENCHMARKS:
            pair_idx += 1
            print()
            _divider('═')
            print(f"  {_c(f'PAIR {pair_idx}/{n_pairs}', _BOLD)}  "
                  f"{_c(model_id, _CYAN)}  ×  {_c(benchmark_id, _CYAN)}")
            _divider('═')

            record = run_pair(model_id, benchmark_id, args.output, args.probe_only)
            results.append(record)

    print_summary(results)
    print(f"Full results written to: {args.output}\n")


if __name__ == '__main__':
    main()


