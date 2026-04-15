"""
Memory Flight Report
====================
Runs a pre-flight memory check then scores the model, tracking actual RSS
throughout so you can compare the estimate against what really happened.

Usage
-----
    python scripts/memory_flight_report.py <model_id> <benchmark_id>

Example
-------
    python scripts/memory_flight_report.py resnet50_tutorial MajajHong2015.IT-pls

Output
------
    ┌─ PRE-FLIGHT ESTIMATE ──────────────────────────────────────────┐
    │  Stimuli:    2560   Features: 200,704   Timebins: 1            │
    │  Activation: 1.91 GB  (×6 overhead → 11.47 GB estimated)       │
    │  Available RAM: 13.6 GB   →  OK                                │
    └────────────────────────────────────────────────────────────────┘
    [scoring runs...]
    ┌─ ACTUAL USAGE ─────────────────────────────────────────────────┐
    │  Baseline RSS:   1.2 GB                                        │
    │  Peak RSS:       4.7 GB   (Δ +3.5 GB)                         │
    │  Final RSS:      2.1 GB   (Δ +0.9 GB)                         │
    │  Estimated:     11.5 GB   →  estimate was ACCURATE (1.2×)      │
    └────────────────────────────────────────────────────────────────┘
"""

import os
import sys
import threading
import time
import argparse
import logging

import psutil

# ---------------------------------------------------------------------------
# Resolve local repos so the script works without installation
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_vision_root = os.path.dirname(_script_dir)
_core_root = os.path.join(os.path.dirname(_vision_root), 'core')
for _p in [_vision_root, _core_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from brainscore_vision import load_model, load_benchmark
from brainscore_vision.benchmark_helpers.memory import preallocate_memory
from brainscore_core.benchmarks import score_benchmark

logging.basicConfig(level=logging.WARNING)

_RESET  = '\033[0m'
_BOLD   = '\033[1m'
_GREEN  = '\033[32m'
_YELLOW = '\033[33m'
_RED    = '\033[31m'
_CYAN   = '\033[36m'


# ---------------------------------------------------------------------------
# Peak RSS monitor (background thread)
# ---------------------------------------------------------------------------

class _PeakMonitor:
    """Polls process RSS every `interval` seconds and records the peak."""

    def __init__(self, interval: float = 0.5):
        self._proc = psutil.Process(os.getpid())
        self._interval = interval
        self._peak = self._proc.memory_info().rss
        self._stop = threading.Event()
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
            self._stop.wait(self._interval)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.2f} GB"


def _ratio_label(estimate_gb: float, actual_delta_gb: float) -> str:
    if actual_delta_gb <= 0:
        return f"{_GREEN}estimate unavailable (no measurable delta){_RESET}"
    ratio = estimate_gb / actual_delta_gb
    if ratio >= 0.8:
        colour = _GREEN
        verdict = f"estimate was ACCURATE ({ratio:.1f}×)"
    elif ratio >= 0.4:
        colour = _YELLOW
        verdict = f"estimate was UNDER by {1/ratio:.1f}×"
    else:
        colour = _RED
        verdict = f"estimate was UNDER by {actual_delta_gb/estimate_gb:.1f}×"
    return f"{colour}{verdict}{_RESET}"


def _box(title: str, lines: list[str], width: int = 66) -> str:
    top    = f"┌─ {_BOLD}{title}{_RESET} " + "─" * (width - len(title) - 3) + "┐"
    bottom = "└" + "─" * (width) + "┘"
    body   = "\n".join(f"│  {l:<{width - 2}}│" for l in lines)
    return f"{top}\n{body}\n{bottom}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Memory flight report for a brainscore scoring run.")
    parser.add_argument("model_identifier")
    parser.add_argument("benchmark_identifier")
    parser.add_argument("--skip-score", action="store_true",
                        help="Only run the pre-flight estimate, skip the actual benchmark.")
    args = parser.parse_args()

    proc = psutil.Process(os.getpid())

    # ------------------------------------------------------------------ #
    #  1. Load model + benchmark                                          #
    # ------------------------------------------------------------------ #
    print(f"\n{_CYAN}Loading model '{args.model_identifier}'...{_RESET}")
    model = load_model(args.model_identifier)

    print(f"{_CYAN}Loading benchmark '{args.benchmark_identifier}'...{_RESET}\n")
    benchmark = load_benchmark(args.benchmark_identifier)

    # ------------------------------------------------------------------ #
    #  2. Pre-flight estimate                                             #
    # ------------------------------------------------------------------ #
    print(f"{_BOLD}Running pre-flight probe (1 stimulus)...{_RESET}")
    try:
        estimate = preallocate_memory(model, benchmark, raise_if_oom=False)
    except TypeError as e:
        print(f"{_YELLOW}Pre-flight skipped (benchmark type not supported): {e}{_RESET}\n")
        estimate = None

    if estimate is not None:
        ok_or_oom = f"{_RED}OOM LIKELY{_RESET}" if estimate.will_oom else f"{_GREEN}OK{_RESET}"
        preflight_lines = [
            f"Stimuli: {estimate.num_stimuli:>6}   Features: {estimate.num_features:>7,}   Timebins: {estimate.num_timebins}",
            f"Activation: {estimate.activation_gb:.2f} GB  "
            f"(×6 overhead → {estimate.total_estimated_gb:.2f} GB estimated)",
            f"Available RAM: {estimate.available_gb:.1f} GB   →  {ok_or_oom}",
        ]
        print(_box("PRE-FLIGHT ESTIMATE", preflight_lines))
        print()

        if estimate.will_oom and not args.skip_score:
            print(f"{_RED}OOM predicted — proceeding anyway to measure actual usage.{_RESET}\n")

    if args.skip_score:
        return

    # ------------------------------------------------------------------ #
    #  3. Score (with RSS monitoring)                                     #
    # ------------------------------------------------------------------ #
    baseline_rss = proc.memory_info().rss
    print(f"{_CYAN}Baseline RSS: {_gb(baseline_rss)}{_RESET}")
    print(f"{_BOLD}Scoring...{_RESET}  (this may take a while)\n")

    monitor = _PeakMonitor(interval=0.5).start()
    t0 = time.time()

    try:
        score = benchmark(model)
        elapsed = time.time() - t0
        peak_rss = monitor.stop()
        final_rss = proc.memory_info().rss
    except AssertionError as e:
        monitor.stop()
        elapsed = time.time() - t0
        print(f"\n{_RED}AssertionError in attach_stimulus_set_meta after {elapsed:.1f}s{_RESET}")
        print("This usually means the activations cache has stale paths.")
        print()
        # Print diagnostic: which paths are mismatching
        try:
            from brainscore_vision.model_helpers.activations.core import lstrip_local
            import numpy as np
            stimulus_set = benchmark._assembly.stimulus_set
            from brainscore_vision.benchmark_helpers.screen import place_on_screen
            ss = place_on_screen(stimulus_set, target_visual_degrees=model.visual_degrees(),
                                 source_visual_degrees=benchmark._visual_degrees)
            expected_paths = [lstrip_local(str(ss.get_stimulus(sid))) for sid in ss['stimulus_id'].values[:3]]
            print(f"{_CYAN}Expected paths (first 3):{_RESET}")
            for p in expected_paths:
                print(f"  {p}")
            # Show what fresh _from_paths returns for comparison
            _am = model.activations_model
            lm = model.layer_model._layer_model
            layer = list(dict.items(lm.region_layer_map))[0][1]
            layer = layer if isinstance(layer, str) else layer[0]
            dummy = _am._extractor._from_paths([str(ss.get_stimulus(ss['stimulus_id'].values[0]))], layers=[layer])
            got_paths = [lstrip_local(p) for p in dummy['stimulus_path'].values[:3]]
            print(f"{_CYAN}Fresh _from_paths result paths (first 3):{_RESET}")
            for p in got_paths:
                print(f"  {p}")
        except Exception as diag_err:
            print(f"(diagnostic failed: {diag_err})")
        print()
        print(f"Fix: delete the stale cache entry and re-run:")
        try:
            cache_dir = os.path.expanduser(
                "~/.result_caching/brainscore_vision.model_helpers.activations.core"
                ".ActivationsExtractorHelper._from_paths_stored"
            )
            cache_file = (
                f"identifier={model.identifier},"
                f"stimuli_identifier={ss.identifier},"
                f"number_of_trials=1,require_variance=False.pkl"
            )
            print(f"  rm '{os.path.join(cache_dir, cache_file)}'")
        except Exception:
            print(
                "  rm ~/.result_caching/brainscore_vision.model_helpers.activations.core"
                ".ActivationsExtractorHelper._from_paths_stored/<model>,<benchmark>*.pkl"
            )
        sys.exit(1)

    except MemoryError as e:
        monitor.stop()
        elapsed = time.time() - t0
        peak_rss = proc.memory_info().rss
        print(f"\n{_RED}MemoryError after {elapsed:.1f}s:{_RESET} {e}")
        print(f"Peak RSS before crash: {_gb(peak_rss)}  (Δ +{_gb(peak_rss - baseline_rss)})\n")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  4. Report                                                          #
    # ------------------------------------------------------------------ #
    delta_peak  = peak_rss  - baseline_rss
    delta_final = final_rss - baseline_rss
    est_gb = estimate.total_estimated_gb if estimate else float('nan')

    actual_lines = [
        f"Baseline RSS:  {_gb(baseline_rss)}",
        f"Peak RSS:      {_gb(peak_rss)}   (Δ +{_gb(delta_peak)})",
        f"Final RSS:     {_gb(final_rss)}   (Δ +{_gb(delta_final)})",
        f"Estimated:     {est_gb:.2f} GB   →  {_ratio_label(est_gb, delta_peak / (1024**3))}",
        f"Elapsed:       {elapsed:.1f}s",
        f"Score:         {float(score):.4f}",
    ]
    print(_box("ACTUAL USAGE", actual_lines))
    print()


if __name__ == '__main__':
    main()
