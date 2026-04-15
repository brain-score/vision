"""
Memory Profile Suite  —  5 × 5 pre-flight analysis
====================================================
Runs the pre-flight memory probe for every (model, benchmark) pair and
displays a colour-coded table showing estimated GB and OOM status.
No full scoring is performed; only the single-stimulus probe runs.

Usage
-----
    python scripts/mem_profile_suite.py [--csv out.csv]

Output example
--------------
    ┌─────────────────────────────────────┬──────────────────────┬── ...
    │ Model                               │ MajajHong2015.IT-pls │ ...
    ├─────────────────────────────────────┼──────────────────────┼── ...
    │ resnet50_tutorial                   │  1.91 GB  ✓          │ ...
    │ alexnet                             │  0.12 GB  ✓          │ ...
    │ vit_large_patch14_clip_224:openai…  │  6.30 GB  ✓          │ ...
    │ VOneCORnet-S                        │  2.84 GB  ✓          │ ...
    │ efficientnet_b0                     │  0.54 GB  ✓          │ ...
    └─────────────────────────────────────┴──────────────────────┴── ...
"""

import os
import sys
import time
import argparse
import csv
import io
import logging

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
    'Marques2020_FreemanZiemba2013-texture_modulation_index',
    'Allen2022_fmri.IT-ridge',
]

# Friendly short names for table columns (≤18 chars)
_BM_SHORT = {
    'MajajHong2015.IT-pls':                                 'MajajHong.IT',
    'Sanghavi2020.IT-pls':                                  'Sanghavi.IT',
    'Papale2025.IT-ridgecv':                                'Papale25.IT',
    'Marques2020_FreemanZiemba2013-texture_modulation_index': 'Marq/FZ-texmod',
    'Allen2022_fmri.IT-ridge':                              'Allen22.IT',
}

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


def _c(text, colour):
    return f"{colour}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Result dataclass (plain dict avoids dataclass import issues)
# ---------------------------------------------------------------------------

def _make_result(model_id, benchmark_id, **kw):
    return dict(model_id=model_id, benchmark_id=benchmark_id, **kw)


# ---------------------------------------------------------------------------
# Probe one (model, benchmark) pair
# ---------------------------------------------------------------------------

def probe_pair(model, model_id, benchmark, benchmark_id):
    """
    Returns a result dict with keys:
        status  : 'ok' | 'oom' | 'skip' | 'error'
        est_gb  : float or None
        feat    : int or None
        stimuli : int or None
        timebins: int or None
        elapsed : float
        note    : str
    """
    from brainscore_vision.benchmark_helpers.memory import preallocate_memory

    t0 = time.time()
    try:
        est = preallocate_memory(model, benchmark, raise_if_oom=False)
    except TypeError as e:
        return _make_result(model_id, benchmark_id,
                            status='skip', est_gb=None, feat=None,
                            stimuli=None, timebins=None,
                            elapsed=time.time() - t0,
                            note=f"unsupported benchmark type: {e}")
    except Exception as e:
        return _make_result(model_id, benchmark_id,
                            status='error', est_gb=None, feat=None,
                            stimuli=None, timebins=None,
                            elapsed=time.time() - t0,
                            note=str(e)[:120])

    elapsed = time.time() - t0
    if est is None:
        return _make_result(model_id, benchmark_id,
                            status='skip', est_gb=None, feat=None,
                            stimuli=None, timebins=None,
                            elapsed=elapsed,
                            note='BRAINSCORE_SKIP_MEMORY_CHECK set')

    return _make_result(model_id, benchmark_id,
                        status='oom' if est.will_oom else 'ok',
                        est_gb=est.total_estimated_gb,
                        feat=est.num_features,
                        stimuli=est.num_stimuli,
                        timebins=est.num_timebins,
                        elapsed=elapsed,
                        note='')


# ---------------------------------------------------------------------------
# Load helpers with error capture
# ---------------------------------------------------------------------------

def _load_model(mid):
    from brainscore_vision import load_model
    return load_model(mid)


def _load_benchmark(bid):
    from brainscore_vision import load_benchmark
    return load_benchmark(bid)


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

_MODEL_W = 38      # width of model name column
_CELL_W  = 17      # width of each benchmark column

def _trunc(s, n):
    return s if len(s) <= n else s[:n - 1] + '…'


def _cell_text(r):
    """Plain text for a result cell (no ANSI)."""
    if r['status'] == 'ok':
        return f"{r['est_gb']:6.2f} GB  OK "
    elif r['status'] == 'oom':
        return f"{r['est_gb']:6.2f} GB OOM"
    elif r['status'] == 'skip':
        return "  —  skip    "
    else:
        return "  —  error   "


def _cell_colour(r):
    """Coloured cell string."""
    if r['status'] == 'ok':
        return _c(f"{r['est_gb']:6.2f} GB", _GREEN) + "  ✓"
    elif r['status'] == 'oom':
        return _c(f"{r['est_gb']:6.2f} GB", _RED) + " OOM"
    elif r['status'] == 'skip':
        return _c("  —  skip", _DIM)
    else:
        return _c("  —  ERR ", _YELLOW)


def _hline(char_mid, char_left, char_right, char_sep):
    parts = [char_left]
    parts.append(char_mid * (_MODEL_W + 2))
    for _ in BENCHMARKS:
        parts.append(char_sep)
        parts.append(char_mid * (_CELL_W + 2))
    parts.append(char_right)
    return ''.join(parts)


def print_table(results_grid):
    """results_grid[model_id][benchmark_id] = result dict"""

    top    = _hline('─', '┌', '┐', '┬')
    mid    = _hline('─', '├', '┤', '┼')
    bottom = _hline('─', '└', '┘', '┴')

    # Header
    print(top)
    header = f"│ {_c(_trunc('Model', _MODEL_W), _BOLD):<{_MODEL_W + len(_BOLD) + len(_RESET)}} "
    for bid in BENCHMARKS:
        short = _BM_SHORT.get(bid, bid)
        header += f"│ {_c(_trunc(short, _CELL_W), _BOLD):<{_CELL_W + len(_BOLD) + len(_RESET)}} "
    header += "│"
    print(header)
    print(mid)

    # Rows
    for mid_id in MODELS:
        row = f"│ {_trunc(mid_id, _MODEL_W):<{_MODEL_W}} "
        for bid in BENCHMARKS:
            r = results_grid[mid_id][bid]
            cell = _cell_colour(r)
            # Pad to fixed visual width (ANSI codes don't take visual space)
            visual_len = len(_cell_text(r))
            padding = _CELL_W - visual_len
            row += f"│ {cell}{' ' * padding} "
        row += "│"
        print(row)

    print(bottom)


# ---------------------------------------------------------------------------
# Detail rows (features / stimuli / time)
# ---------------------------------------------------------------------------

def print_details(results_grid):
    print(f"\n{_c('Details  (features × stimuli × timebins,  probe elapsed)', _DIM)}")
    print()
    for mid_id in MODELS:
        print(f"  {_c(_trunc(mid_id, 45), _BOLD)}")
        for bid in BENCHMARKS:
            r = results_grid[mid_id][bid]
            short = _BM_SHORT.get(bid, bid)
            if r['status'] in ('ok', 'oom'):
                line = (f"    {short:<18}  "
                        f"{r['feat']:>7,} feat × {r['stimuli']:>5,} stimuli "
                        f"× {r['timebins']} tbin   "
                        f"{r['est_gb']:.2f} GB total   "
                        f"probe {r['elapsed']:.1f}s")
            elif r['status'] == 'skip':
                line = f"    {short:<18}  skipped ({r['note']})"
            else:
                line = f"    {short:<18}  ERROR: {r['note']}"
            print(line)
        print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(results_grid, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'benchmark', 'status', 'total_est_gb',
                    'num_features', 'num_stimuli', 'num_timebins',
                    'probe_elapsed_s', 'note'])
        for mid_id in MODELS:
            for bid in BENCHMARKS:
                r = results_grid[mid_id][bid]
                w.writerow([
                    mid_id, bid,
                    r['status'],
                    f"{r['est_gb']:.4f}" if r['est_gb'] is not None else '',
                    r['feat'] or '',
                    r['stimuli'] or '',
                    r['timebins'] or '',
                    f"{r['elapsed']:.2f}",
                    r['note'],
                ])
    print(f"\n{_c('CSV written →', _CYAN)} {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="5×5 pre-flight memory profile for brainscore_vision models × benchmarks.")
    parser.add_argument('--csv', metavar='PATH', default=None,
                        help='also write results to a CSV file')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Load all benchmarks first (shared across models)                  #
    # ------------------------------------------------------------------ #
    print(f"\n{_c('Loading benchmarks...', _CYAN)}")
    benchmarks = {}
    for bid in BENCHMARKS:
        sys.stdout.write(f"  {bid}... ")
        sys.stdout.flush()
        t0 = time.time()
        try:
            benchmarks[bid] = _load_benchmark(bid)
            print(f"{_c('OK', _GREEN)} ({time.time()-t0:.1f}s)")
        except Exception as e:
            benchmarks[bid] = None
            print(f"{_c('FAILED', _RED)}: {str(e)[:80]}")

    # ------------------------------------------------------------------ #
    #  For each model, probe against all benchmarks                      #
    # ------------------------------------------------------------------ #
    results_grid = {mid_id: {} for mid_id in MODELS}

    for mid_id in MODELS:
        print(f"\n{_c(f'Loading model: {mid_id}', _CYAN)}")
        t0 = time.time()
        try:
            model = _load_model(mid_id)
            print(f"  {_c('OK', _GREEN)} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  {_c('FAILED', _RED)}: {str(e)[:80]}")
            # Fill all benchmark slots with error
            for bid in BENCHMARKS:
                results_grid[mid_id][bid] = _make_result(
                    mid_id, bid, status='error', est_gb=None,
                    feat=None, stimuli=None, timebins=None,
                    elapsed=0.0, note=f"model load failed: {str(e)[:60]}")
            continue

        for bid in BENCHMARKS:
            bm = benchmarks.get(bid)
            if bm is None:
                results_grid[mid_id][bid] = _make_result(
                    mid_id, bid, status='error', est_gb=None,
                    feat=None, stimuli=None, timebins=None,
                    elapsed=0.0, note='benchmark failed to load')
                continue

            sys.stdout.write(f"  probing {_BM_SHORT.get(bid, bid):<18}... ")
            sys.stdout.flush()
            r = probe_pair(model, mid_id, bm, bid)
            results_grid[mid_id][bid] = r

            if r['status'] == 'ok':
                print(f"{_c('OK', _GREEN)}  {r['est_gb']:.2f} GB  ({r['elapsed']:.1f}s)")
            elif r['status'] == 'oom':
                print(f"{_c('OOM', _RED)}  {r['est_gb']:.2f} GB  ({r['elapsed']:.1f}s)")
            elif r['status'] == 'skip':
                print(f"{_c('skip', _DIM)}  ({r['elapsed']:.1f}s)")
            else:
                print(f"{_c('error', _YELLOW)}  {r['note'][:60]}")

    # ------------------------------------------------------------------ #
    #  Print table + details                                              #
    # ------------------------------------------------------------------ #
    print(f"\n\n{'═' * 70}")
    print(f"  {_c('MEMORY PROFILE MATRIX', _BOLD)}  "
          f"(total estimated GB = activations × {_c('6×', _DIM)} overhead)")
    print(f"{'═' * 70}\n")
    print_table(results_grid)
    print_details(results_grid)

    if args.csv:
        write_csv(results_grid, args.csv)


if __name__ == '__main__':
    main()
