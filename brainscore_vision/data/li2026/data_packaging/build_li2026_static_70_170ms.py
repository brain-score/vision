"""Build the production static Li2026 assembly: a fixed 70-170 ms post-onset window
applied uniformly to every unit (MajajHong convention), with a window-matched noise
ceiling. Run after build_li2026_static.py (for metadata), build_li2026_temporal.py
(for the response), and build_li2026_reliability_70_170ms.py (for the window ceiling);
then package_li2026_static_70_170ms.py uploads it.

Why a fixed window (vs the upstream per-unit `response_best`)
-------------------------------------------------------------
The per-unit best window is right for Li et al.'s unit-characterization analyses, but
Brain-Score's other primate-IT static benchmarks (MajajHong2015, Sanghavi2020) use a
single fixed window applied to every unit. Mixing conventions makes leaderboard scores
non-comparable and inflates Li2026 via per-unit window-selection bias. The paper itself
uses fixed/binned windows for its cross-population analyses (Fig 4 RSA, Fig 5 encoding).

What this builds
----------------
* response          : mean firing rate over the ten 10-ms temporal bins covering 70-170 ms.
* reliability       : paper-canonical best-window split-half SB (kept for provenance and to
                      reproduce the paper's reliable-unit counts; from the original static).
* reliability_window: split-half SB recomputed AT 70-170 ms -- the noise ceiling matching the
                      scored response. The benchmark selects and ceils on this coord.
* arealabel/snr/best_time_start/best_time_end/tn_index: merged back from the original static
                      (the temporal assembly does not carry them).

Inputs (under --build-dir):
  Li2026_temporal_assembly.nc       (presentation x neuroid x 30 bins @ 10 ms)
  Li2026_static_assembly.nc         (original best-window static -- source of metadata coords)
  Li2026_reliability_70_170ms.csv   (neuroid_id, reliability_window)
Output:
  Li2026_static_70_170ms_assembly.nc
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

WINDOW_START_MS, WINDOW_END_MS = 70, 170
METADATA_COORDS = ["arealabel", "snr", "best_time_start", "best_time_end", "tn_index"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default="/Volumes/Hagibis/triple-n/build")
    args = ap.parse_args()
    bd = Path(args.build_dir)

    T = xr.open_dataset(str(bd / "Li2026_temporal_assembly.nc"))["__xarray_dataarray_variable__"]
    mask = (T.time_bin_start.values >= WINDOW_START_MS) & (T.time_bin_end.values <= WINDOW_END_MS)
    assert mask.sum() == 10, f"expected ten 10-ms bins in 70-170 ms, got {mask.sum()}"
    resp = T.isel(time_bin=np.where(mask)[0]).mean("time_bin")          # (presentation, neuroid)
    nid = [str(x) for x in resp["neuroid_id"].values]

    # metadata coords merged from the original best-window static, aligned by neuroid_id
    orig = xr.open_dataarray(str(bd / "Li2026_static_assembly.nc")).squeeze("time_bin")
    o_pos = {str(n): i for i, n in enumerate(orig["neuroid_id"].values)}
    assert set(nid) == set(o_pos), "neuroid_id mismatch between temporal and original static"
    order = np.array([o_pos[n] for n in nid])
    extra = {c: ("neuroid", orig[c].values[order]) for c in METADATA_COORDS
             if c in orig.coords and orig[c].dims == ("neuroid",)}

    # window-matched reliability merged from the recompute CSV
    rel_win = (pd.read_csv(bd / "Li2026_reliability_70_170ms.csv", dtype={"neuroid_id": str})
               .set_index("neuroid_id")["reliability_window"].reindex(nid).values.astype(np.float32))
    print(f"reliability_window: matched {np.isfinite(rel_win).sum()}/{len(nid)}; "
          f"median={np.nanmedian(rel_win):.3f}; >0.4={int((rel_win > 0.4).sum())}")

    coords = {c: (resp[c].dims, resp[c].values) for c in resp.coords
              if resp[c].dims in (("neuroid",), ("presentation",))}
    coords.update(extra)
    coords["reliability_window"] = ("neuroid", rel_win)
    coords["time_bin_start"] = ("time_bin", [WINDOW_START_MS])
    coords["time_bin_end"] = ("time_bin", [WINDOW_END_MS])
    static = xr.DataArray(resp.values[:, :, None], dims=("presentation", "neuroid", "time_bin"),
                          coords=coords)
    static.attrs.update({"source_assembly": "Li2026_temporal_Assembly",
                         "window_start_ms": WINDOW_START_MS, "window_end_ms": WINDOW_END_MS,
                         "ceiling_coord": "reliability_window (split-half SB at 70-170 ms)",
                         "selection_coord": "reliability_window > 0.4"})

    reg = static["region"].values
    print(f"static dims: {dict(static.sizes)}  coords: {sorted(static.coords)}")
    for r in ("V1", "V2", "V4", "IT"):
        print(f"  {r}: reliable(70-170ms)={int(((reg == r) & (rel_win > 0.4)).sum())}  "
              f"(best-window={int(((reg == r) & (static['reliability'].values > 0.4)).sum())})")
    out = bd / "Li2026_static_70_170ms_assembly.nc"
    static.to_netcdf(str(out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
