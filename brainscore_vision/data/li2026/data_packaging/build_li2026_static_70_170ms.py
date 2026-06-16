"""Derive the canonical Brain-Score static Li2026 assembly from the temporal
assembly by averaging firing rates within a fixed 70-170 ms post-onset window.

Why this exists
---------------
The original Li2026 static assembly (``build_li2026_static.py``) was built
from the upstream ``response_best`` matrix, which uses a *per-unit* best
window selected via the paper's moving-window approach. That convention is
appropriate for the paper's unit-characterization analyses (reliability,
SNR, selectivity) and is cross-validated for those, but it is **not** the
Brain-Score convention for model-vs-brain population scoring:

  * MajajHong2015, Sanghavi2020, and the other primate-IT static benchmarks
    all use a fixed population-level window (70-170 ms post-onset) applied
    uniformly to every unit.
  * The paper itself falls back to fixed/binned windows for its
    cross-population analyses (Fig 4 RSA uses 20 ms bins; Fig 5 encoding
    models use a fixed peak time-lag).

Rebuilding the static from the temporal assembly with a 70-170 ms window
brings Li2026 in line with the rest of the suite, so its leaderboard
scores are directly comparable to MajajHong/Sanghavi/etc.

Inputs / outputs
----------------
* Input:  ``/Volumes/Hagibis/triple-n/build/Li2026_temporal_assembly.nc``
         (presentation=1000 x neuroid=47,503 x time_bin=30 @ 10 ms bins, 0-300 ms)
* Output: ``/Volumes/Hagibis/triple-n/build/Li2026_static_70_170ms_assembly.nc``
         (presentation=1000 x neuroid=47,503 x time_bin=1, start=70 end=170)

Empirical justification (logged at run-time):
The reliable-IT population PSTH peaks at ~130-140 ms with the >50%-of-peak
band spanning 100-220 ms. A 70-170 ms window covers the leading edge
through peak — same shape as the MajajHong convention.
"""
from pathlib import Path
import numpy as np
import xarray as xr

BUILD_DIR = Path("/Volumes/Hagibis/triple-n/build")
SOURCE_NC = BUILD_DIR / "Li2026_temporal_assembly.nc"
TARGET_NC = BUILD_DIR / "Li2026_static_70_170ms_assembly.nc"

WINDOW_START_MS = 70
WINDOW_END_MS = 170


def main() -> None:
    ds = xr.open_dataset(str(SOURCE_NC))
    da = ds["__xarray_dataarray_variable__"]
    print(f"loaded {SOURCE_NC.name}: dims={dict(da.sizes)}")

    # Select bins whose [start, end) falls inside [70, 170). The temporal
    # assembly's 10 ms grid makes this exactly 10 bins (70-80, ..., 160-170).
    mask = (ds.time_bin_start.values >= WINDOW_START_MS) & (ds.time_bin_end.values <= WINDOW_END_MS)
    selected_starts = ds.time_bin_start.values[mask]
    print(f"selected {mask.sum()} bins: starts={selected_starts.tolist()}")
    assert mask.sum() == 10, "expected exactly ten 10-ms bins covering 70-170 ms"

    # Mean firing rate over the window (neurons share the same window — the
    # whole point of this rebuild vs response_best). Preserves all neuroid
    # and presentation coords automatically; time_bin coord is collapsed.
    static = da.isel(time_bin=np.where(mask)[0]).mean("time_bin")

    # Re-attach a single-bin time axis matching the MajajHong static format
    # (assemblies in that benchmark keep time_bin=1 with start/end coords so
    # consumers can introspect the window). Adds the dim back without
    # broadcasting the data — just a singleton axis with the start/end pair.
    # Transpose to (presentation, neuroid, time_bin) to match the prior
    # Li2026 static dim order (and MajajHong's static layout) so downstream
    # code that hardcodes axis order doesn't have to be touched.
    static = static.expand_dims(time_bin=[0])
    static = static.assign_coords({
        "time_bin_start": ("time_bin", [WINDOW_START_MS]),
        "time_bin_end": ("time_bin", [WINDOW_END_MS]),
    })
    static = static.transpose("presentation", "neuroid", "time_bin")

    # Record provenance in attrs so a future maintainer doesn't have to
    # reverse-engineer the build from the file size.
    static.attrs.update({
        "source_assembly": "Li2026_temporal_Assembly",
        "window_start_ms": WINDOW_START_MS,
        "window_end_ms": WINDOW_END_MS,
        "window_bins_averaged": int(mask.sum()),
        "build_script": "build_li2026_static_70_170ms.py",
        "notes": (
            "Mean firing rate in a fixed 70-170 ms post-onset window applied "
            "uniformly to all units. Replaces the prior per-unit best-window "
            "static (`response_best`) so the assembly matches the MajajHong "
            "convention and is comparable to other Brain-Score primate-IT "
            "benchmarks. Unit selection (reliability_best > 0.4) is unchanged."
        ),
    })

    print(f"output dims: {dict(static.sizes)}")
    print(f"writing {TARGET_NC}")
    static.to_netcdf(str(TARGET_NC))
    print("done.")


if __name__ == "__main__":
    main()
