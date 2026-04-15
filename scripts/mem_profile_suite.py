"""
Memory Profile Suite  —  5 × 5 pre-flight analysis
====================================================
Runs the pre-flight memory probe for every (model, benchmark) pair and
displays a colour-coded table showing estimated GB and OOM status.
No full scoring is performed; only the single-stimulus probe runs.

Usage
-----
    python scripts/mem_profile_suite.py [--csv out.csv]
"""

import os
import sys
import time
import argparse
import csv
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

_BM_SHORT = {
    'MajajHong2015.IT-pls':                                   'MajajHong.IT',
    'Sanghavi2020.IT-pls':                                    'Sanghavi.IT',
    'Papale2025.IT-ridgecv':                                  'Papale25.IT',
    'Marques2020_FreemanZiemba2013-texture_modulation_index': 'Marq/FZ-texmod',
    'Allen2022_fmri.IT-ridge':                                'Allen22.IT',
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
_BLUE   = '\033[34m'


def _c(text, colour):
    return f"{colour}{text}{_RESET}"


def _step(msg, indent=4):
    """Print a progress sub-step with a right-arrow prefix."""
    print(f"{' ' * indent}{_c('→', _BLUE)} {msg}", flush=True)


def _substep(msg, indent=6):
    """Print a nested sub-step."""
    print(f"{' ' * indent}{_c('·', _DIM)} {msg}", flush=True)


def _ok(msg=''):
    return _c(f'OK{(" " + msg) if msg else ""}', _GREEN)


def _fail(msg=''):
    return _c(f'FAIL{(" " + msg) if msg else ""}', _RED)


# ---------------------------------------------------------------------------
# Benchmark metadata inspector
# ---------------------------------------------------------------------------

def _inspect_benchmark(bm):
    """Return a dict of human-readable benchmark metadata, safely."""
    info = {}
    for attr in ('region', 'timebins', '_visual_degrees', '_number_of_trials'):
        try:
            info[attr] = getattr(bm, attr)
        except Exception:
            info[attr] = '?'

    # stimulus count
    try:
        from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, TrainTestNeuralBenchmark
        if isinstance(bm, NeuralBenchmark):
            info['num_stimuli'] = int(bm._assembly.stimulus_set['stimulus_id'].nunique())
            info['bm_type'] = 'NeuralBenchmark'
        elif isinstance(bm, TrainTestNeuralBenchmark):
            n_train = int(bm.train_assembly.stimulus_set['stimulus_id'].nunique())
            n_test  = int(bm.test_assembly.stimulus_set['stimulus_id'].nunique())
            info['num_stimuli'] = n_train + n_test
            info['n_train'] = n_train
            info['n_test'] = n_test
            info['bm_type'] = 'TrainTestNeuralBenchmark'
        else:
            info['num_stimuli'] = '?'
            info['bm_type'] = type(bm).__name__
    except Exception as e:
        info['num_stimuli'] = f'? ({e})'
        info['bm_type'] = type(bm).__name__

    return info


# ---------------------------------------------------------------------------
# Model metadata inspector
# ---------------------------------------------------------------------------

def _inspect_model(model):
    """Return a dict of human-readable model metadata, safely."""
    info = {}
    try:
        info['visual_degrees'] = model.visual_degrees()
    except Exception:
        info['visual_degrees'] = '?'

    # Try to read committed layer map
    try:
        lm = getattr(model, 'layer_model', None)
        if lm is not None and hasattr(lm, '_layer_model'):
            lm = lm._layer_model
        rmap = getattr(lm, 'region_layer_map', None) if lm else None
        if rmap is None:
            rmap = getattr(model, 'region_layer_map', None)
        if rmap is not None:
            # Safely read committed entries without triggering lazy RegionLayerMap
            committed = {}
            for region in ['V1', 'V2', 'V4', 'IT']:
                if dict.__contains__(rmap, region):
                    val = dict.__getitem__(rmap, region)
                    committed[region] = val if isinstance(val, str) else (val[0] if val else '?')
            info['region_layer_map'] = committed
        else:
            info['region_layer_map'] = {}
    except Exception as e:
        info['region_layer_map'] = {'error': str(e)[:40]}

    # activations model identifier
    try:
        info['activations_model'] = model.activations_model.identifier
    except Exception:
        info['activations_model'] = '?'

    return info


# ---------------------------------------------------------------------------
# Result helper
# ---------------------------------------------------------------------------

def _make_result(model_id, benchmark_id, **kw):
    return dict(model_id=model_id, benchmark_id=benchmark_id, **kw)


# ---------------------------------------------------------------------------
# Probe one (model, benchmark) pair  —  with verbose step printing
# ---------------------------------------------------------------------------

def probe_pair(model, model_id, benchmark, benchmark_id):
    from brainscore_vision.benchmark_helpers.memory import (
        preallocate_memory, _get_probe_layer, _OVERHEAD_FACTOR, _BYTES_PER_ELEMENT,
    )
    from brainscore_vision.benchmark_helpers.neural_common import NeuralBenchmark, TrainTestNeuralBenchmark
    from brainscore_vision.benchmark_helpers.screen import place_on_screen

    short = _BM_SHORT.get(benchmark_id, benchmark_id)
    ind = 6  # indentation for sub-steps under the probe

    # ── Step 1: benchmark metadata ──────────────────────────────────────
    _step(f"benchmark metadata", indent=4)
    try:
        if isinstance(benchmark, NeuralBenchmark):
            ss = benchmark._assembly.stimulus_set
            n_stimuli = int(ss['stimulus_id'].nunique())
            timebins  = benchmark.timebins
            region    = benchmark.region
            vis_deg   = benchmark._visual_degrees
            _substep(f"type=NeuralBenchmark  region={region}  stimuli={n_stimuli:,}  "
                     f"timebins={len(timebins)}×{timebins[0] if timebins else '?'}  "
                     f"visual_deg={vis_deg}")
        elif isinstance(benchmark, TrainTestNeuralBenchmark):
            n_train = int(benchmark.train_assembly.stimulus_set['stimulus_id'].nunique())
            n_test  = int(benchmark.test_assembly.stimulus_set['stimulus_id'].nunique())
            timebins = benchmark.timebins
            region   = benchmark.region
            vis_deg  = benchmark._visual_degrees
            _substep(f"type=TrainTestNeuralBenchmark  region={region}  "
                     f"train={n_train:,}  test={n_test:,}  total={(n_train+n_test):,}  "
                     f"timebins={len(timebins)}  visual_deg={vis_deg}")
        else:
            _substep(f"type={type(benchmark).__name__}  (unsupported — will skip)")
    except Exception as e:
        _substep(f"metadata read failed: {e}")

    # ── Step 2: model → layer resolution ────────────────────────────────
    _step("layer resolution", indent=4)
    try:
        _am = getattr(model, 'activations_model', None)
        _extractor = getattr(_am, '_extractor', None) if _am else None
        probe_layer = _get_probe_layer(model) if _extractor is not None else None
        if probe_layer:
            _substep(f"probe layer = {_c(probe_layer, _CYAN)}  "
                     f"(via activations_model._extractor._from_paths — no cache write)")
        else:
            _substep(f"no layer found via fast path — will fall back to model.look_at()")
    except Exception as e:
        _substep(f"layer resolution failed: {e}")

    # ── Step 3: place_on_screen check ───────────────────────────────────
    _step("visual-degree check", indent=4)
    try:
        model_deg = model.visual_degrees()
        bench_deg = benchmark._visual_degrees if hasattr(benchmark, '_visual_degrees') else '?'
        if model_deg == bench_deg:
            _substep(f"model={model_deg}°  benchmark={bench_deg}°  → place_on_screen is a no-op")
        else:
            _substep(f"model={model_deg}°  benchmark={bench_deg}°  "
                     f"→ place_on_screen will resize to {model_deg}°")
    except Exception as e:
        _substep(f"degree check failed: {e}")

    # ── Step 4: run the probe ────────────────────────────────────────────
    _step(f"running 1-stimulus forward pass...", indent=4)
    t0 = time.time()
    try:
        est = preallocate_memory(model, benchmark, raise_if_oom=False)
    except TypeError as e:
        elapsed = time.time() - t0
        _substep(_c(f"skipped — unsupported benchmark type: {e}", _DIM))
        return _make_result(model_id, benchmark_id,
                            status='skip', est_gb=None, feat=None,
                            stimuli=None, timebins=None,
                            elapsed=elapsed,
                            note=f"unsupported benchmark type: {e}")
    except Exception as e:
        elapsed = time.time() - t0
        _substep(_c(f"ERROR: {str(e)[:100]}", _RED))
        return _make_result(model_id, benchmark_id,
                            status='error', est_gb=None, feat=None,
                            stimuli=None, timebins=None,
                            elapsed=elapsed,
                            note=str(e)[:120])

    elapsed = time.time() - t0

    if est is None:
        _substep(_c("skipped (BRAINSCORE_SKIP_MEMORY_CHECK set)", _DIM))
        return _make_result(model_id, benchmark_id,
                            status='skip', est_gb=None, feat=None,
                            stimuli=None, timebins=None,
                            elapsed=elapsed,
                            note='BRAINSCORE_SKIP_MEMORY_CHECK set')

    # ── Step 5: results ──────────────────────────────────────────────────
    _step("estimate", indent=4)
    _substep(f"features={est.num_features:,}  stimuli={est.num_stimuli:,}  timebins={est.num_timebins}")
    raw_gb = est.activation_gb
    _substep(f"activation array = {raw_gb:.3f} GB  "
             f"× {_OVERHEAD_FACTOR} overhead = {est.total_estimated_gb:.3f} GB total")
    avail = est.available_gb
    if est.will_oom:
        verdict = _c(f"OOM LIKELY  (need {est.total_estimated_gb:.1f} GB, have {avail:.1f} GB)", _RED)
    else:
        headroom = avail - est.total_estimated_gb
        verdict = _c(f"OK  ({headroom:.1f} GB headroom of {avail:.1f} GB available)", _GREEN)
    _substep(f"verdict: {verdict}")
    _substep(f"probe elapsed: {elapsed:.2f}s")

    return _make_result(model_id, benchmark_id,
                        status='oom' if est.will_oom else 'ok',
                        est_gb=est.total_estimated_gb,
                        feat=est.num_features,
                        stimuli=est.num_stimuli,
                        timebins=est.num_timebins,
                        elapsed=elapsed,
                        note='')


# ---------------------------------------------------------------------------
# Load helpers
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

_MODEL_W = 38
_CELL_W  = 17


def _trunc(s, n):
    return s if len(s) <= n else s[:n - 1] + '…'


def _cell_text(r):
    if r['status'] == 'ok':
        return f"{r['est_gb']:6.2f} GB  OK "
    elif r['status'] == 'oom':
        return f"{r['est_gb']:6.2f} GB OOM"
    elif r['status'] == 'skip':
        return "  —  skip    "
    else:
        return "  —  error   "


def _cell_colour(r):
    if r['status'] == 'ok':
        return _c(f"{r['est_gb']:6.2f} GB", _GREEN) + "  ✓"
    elif r['status'] == 'oom':
        return _c(f"{r['est_gb']:6.2f} GB", _RED) + " OOM"
    elif r['status'] == 'skip':
        return _c("  —  skip", _DIM)
    else:
        return _c("  —  ERR ", _YELLOW)


def _hline(char_mid, char_left, char_right, char_sep):
    parts = [char_left, char_mid * (_MODEL_W + 2)]
    for _ in BENCHMARKS:
        parts += [char_sep, char_mid * (_CELL_W + 2)]
    parts.append(char_right)
    return ''.join(parts)


def print_table(results_grid):
    top    = _hline('─', '┌', '┐', '┬')
    mid    = _hline('─', '├', '┤', '┼')
    bottom = _hline('─', '└', '┘', '┴')

    print(top)
    header = f"│ {_c(_trunc('Model', _MODEL_W), _BOLD):<{_MODEL_W + len(_BOLD) + len(_RESET)}} "
    for bid in BENCHMARKS:
        short = _BM_SHORT.get(bid, bid)
        header += f"│ {_c(_trunc(short, _CELL_W), _BOLD):<{_CELL_W + len(_BOLD) + len(_RESET)}} "
    header += "│"
    print(header)
    print(mid)

    for mid_id in MODELS:
        row = f"│ {_trunc(mid_id, _MODEL_W):<{_MODEL_W}} "
        for bid in BENCHMARKS:
            r = results_grid[mid_id][bid]
            cell = _cell_colour(r)
            padding = _CELL_W - len(_cell_text(r))
            row += f"│ {cell}{' ' * padding} "
        row += "│"
        print(row)

    print(bottom)


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

    n_bm = len(BENCHMARKS)
    n_m  = len(MODELS)
    total_pairs = n_m * n_bm

    print(f"\n{'═' * 66}")
    print(f"  {_c('MEM PROFILE SUITE', _BOLD)}  —  {n_m} models × {n_bm} benchmarks = {total_pairs} probes")
    print(f"{'═' * 66}")

    # ------------------------------------------------------------------ #
    #  Load all benchmarks first (shared across models)                  #
    # ------------------------------------------------------------------ #
    print(f"\n{_c(f'[1/{n_m + 1}] Loading {n_bm} benchmarks', _CYAN)}\n")
    benchmarks = {}
    for i, bid in enumerate(BENCHMARKS, 1):
        short = _BM_SHORT.get(bid, bid)
        print(f"  [{i}/{n_bm}] {bid}")
        t0 = time.time()
        try:
            bm = _load_benchmark(bid)
            benchmarks[bid] = bm
            elapsed = time.time() - t0
            # Print metadata immediately after load
            info = _inspect_benchmark(bm)
            _step(f"{_ok(f'{elapsed:.1f}s')}  type={info.get('bm_type', '?')}")
            stimuli_str = (
                f"train={info['n_train']:,}+test={info['n_test']:,}={info['num_stimuli']:,}"
                if 'n_train' in info else
                f"stimuli={info['num_stimuli']:,}" if isinstance(info.get('num_stimuli'), int) else
                f"stimuli={info.get('num_stimuli','?')}"
            )
            tb = info.get('timebins', [])
            tb_str = f"{len(tb)} timebin(s)" if isinstance(tb, list) else str(tb)
            _step(f"region={info.get('region','?')}  {stimuli_str}  "
                  f"{tb_str}  visual_deg={info.get('_visual_degrees','?')}  "
                  f"trials={info.get('_number_of_trials','?')}")
        except Exception as e:
            benchmarks[bid] = None
            _step(_fail(f"{str(e)[:80]}"))
        print()

    # ------------------------------------------------------------------ #
    #  For each model, probe against all benchmarks                      #
    # ------------------------------------------------------------------ #
    results_grid = {mid_id: {} for mid_id in MODELS}
    pair_num = 0

    for m_idx, mid_id in enumerate(MODELS, 1):
        section = f"[{m_idx + 1}/{n_m + 1}]"
        print(f"\n{'─' * 66}")
        print(f"{_c(f'{section} Model: {mid_id}', _CYAN)}")
        print(f"{'─' * 66}")

        # Load model
        _step("loading model...", indent=2)
        t0 = time.time()
        try:
            model = _load_model(mid_id)
            elapsed = time.time() - t0
            _step(_ok(f"{elapsed:.1f}s"), indent=2)
            # Print model metadata
            info = _inspect_model(model)
            _step(f"visual_degrees={info.get('visual_degrees','?')}  "
                  f"activations_model={info.get('activations_model','?')}", indent=2)
            rlm = info.get('region_layer_map', {})
            if rlm and 'error' not in rlm:
                layer_str = '  '.join(f"{r}→{l}" for r, l in rlm.items())
                _step(f"committed layers: {_c(layer_str, _CYAN)}", indent=2)
            elif 'error' in rlm:
                _step(f"region_layer_map read error: {rlm['error']}", indent=2)
            print()
        except Exception as e:
            elapsed = time.time() - t0
            _step(_fail(f"{str(e)[:80]}"), indent=2)
            for bid in BENCHMARKS:
                results_grid[mid_id][bid] = _make_result(
                    mid_id, bid, status='error', est_gb=None,
                    feat=None, stimuli=None, timebins=None,
                    elapsed=0.0, note=f"model load failed: {str(e)[:60]}")
                pair_num += 1
            print()
            continue

        # Probe against each benchmark
        for b_idx, bid in enumerate(BENCHMARKS, 1):
            pair_num += 1
            short = _BM_SHORT.get(bid, bid)
            print(f"  {_c(f'pair {pair_num}/{total_pairs}', _DIM)}  "
                  f"{_c(mid_id[:24], _BOLD)} × {_c(short, _BOLD)}")

            bm = benchmarks.get(bid)
            if bm is None:
                _step(_fail("benchmark failed to load — skipping"), indent=4)
                results_grid[mid_id][bid] = _make_result(
                    mid_id, bid, status='error', est_gb=None,
                    feat=None, stimuli=None, timebins=None,
                    elapsed=0.0, note='benchmark failed to load')
                print()
                continue

            r = probe_pair(model, mid_id, bm, bid)
            results_grid[mid_id][bid] = r

            # Summary line for this pair
            if r['status'] == 'ok':
                summary = _c(f"✓  {r['est_gb']:.2f} GB total  ({r['elapsed']:.1f}s)", _GREEN)
            elif r['status'] == 'oom':
                summary = _c(f"✗ OOM  {r['est_gb']:.2f} GB total  ({r['elapsed']:.1f}s)", _RED)
            elif r['status'] == 'skip':
                summary = _c(f"— skipped  ({r['elapsed']:.1f}s)", _DIM)
            else:
                summary = _c(f"✗ error  ({r['elapsed']:.1f}s)", _YELLOW)
            _step(summary, indent=4)
            print()

    # ------------------------------------------------------------------ #
    #  Final table                                                        #
    # ------------------------------------------------------------------ #
    print(f"\n\n{'═' * 66}")
    print(f"  {_c('MEMORY PROFILE MATRIX', _BOLD)}  "
          f"(total estimated GB = activations × {_c('6×', _DIM)} overhead)")
    print(f"{'═' * 66}\n")
    print_table(results_grid)
    print_details(results_grid)

    if args.csv:
        write_csv(results_grid, args.csv)


if __name__ == '__main__':
    main()
