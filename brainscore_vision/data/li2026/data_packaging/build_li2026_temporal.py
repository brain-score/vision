"""Build the Li2026 (Triple-N) temporal neural assembly from the raw ScienceDB release.
Run this (and build_li2026_static.py, for the stimulus mapping) before
:mod:`package_li2026_temporal` (which uploads the output to S3).

Raw source -- Triple-N, ScienceDB doi:10.57760/sciencedb.33556::

  <raw-root>/GoodUnit/GoodUnit_<date>_<monkey>_g<n>.mat   per-unit PSTHs (HDF5/v7.3):
        GoodUnitStrc.response_matrix_img[u] -> (450 time x 1072 img) trial-averaged PSTH,
        .spikepos (probe position), .unittype; global_params.pre_onset (=50 ms baseline).
  <raw-root>/Processed/Processed_ses*.mat                 reliability/selectivity + pos,
        used to recover the session index and per-unit reliability for each GoodUnit file.
  <raw-root>/exclude_area.xls                             per-session area table.

GoodUnit files carry no session index, so each is matched to a Processed session by
``(date, unit-count)`` and verified by ``spikepos == pos`` (a g-number is NOT a session
index). Per-unit position/unittype come from GoodUnit; reliability/selectivity from the
matched Processed session. Sessions that fail to match (or are truncated) are skipped.

Stimulus alignment is read from the static build's output so both assemblies share the
exact same Allen2022-aligned stimulus_id order::

  <stim-csv>  (default: <out-dir of build_li2026_static>/Li2026_stimulus_set.csv)

Output (consumed by package_li2026_temporal.py)::

  <out>  Li2026_temporal_assembly.nc   presentation(1000) x neuroid x time_bin(0-300ms @ 10ms)

Region rule matches the static assembly: IT sessions pool every unit as ``IT``; EVC
sessions assign V1/V2/V4 by probe position. Reliability filtering is applied by the
benchmark at load time, not here.
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr

UTYPE = {1: "SU", 2: "MUA", 3: "NonSom"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default="/Volumes/Hagibis/triple-n",
                    help="directory containing GoodUnit/, Processed/ and exclude_area.xls")
    ap.add_argument("--stim-csv", default="/Volumes/Hagibis/triple-n/build/Li2026_stimulus_set.csv")
    ap.add_argument("--out", default="/Volumes/Hagibis/triple-n/build/Li2026_temporal_assembly.nc")
    ap.add_argument("--bin", type=int, default=10, help="time-bin width (ms)")
    ap.add_argument("--tstart", type=int, default=0, help="window start post-onset (ms)")
    ap.add_argument("--tend", type=int, default=300, help="window end post-onset (ms)")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N GoodUnit files (debug)")
    return ap.parse_args()


def session_domain(area: pd.DataFrame) -> dict:
    return {s: ("EVC" if set(g.Area.astype(str).str.strip().str.upper()) == {"EVC"} else "IT")
            for s, g in area.groupby("SesIdx")}


def region_of(area: pd.DataFrame, domain: dict, pos: float, ses: int) -> str:
    """IT sessions pool as IT; EVC sessions resolve V1/V2/V4 by probe position."""
    if domain[ses] == "IT":
        return "IT"
    for _, r in area[area.SesIdx == ses].iterrows():
        if r.y1 < pos < r.y2:
            label = str(r.AREALABEL).strip()
            for v in ("V1", "V2", "V4"):
                if label.startswith(v):
                    return v
    return "EVC-other"


def index_processed(processed_dir: str):
    """Index Processed sessions by (date, unit-count) for GoodUnit matching."""
    by_key, by_date = {}, defaultdict(list)
    for pf in glob.glob(f"{processed_dir}/Processed_ses*.mat"):
        match = re.match(r"Processed_ses(\d+)_(\d+)_M(\d)_(\d+)\.mat", os.path.basename(pf))
        p = sio.loadmat(pf, squeeze_me=True)
        pos = np.atleast_1d(p["pos"]).astype(float)
        rec = dict(ses=int(match.group(1)), date=match.group(2), animal="M" + match.group(3),
                   pos=pos, rel=np.atleast_1d(p["reliability_best"]).astype(float),
                   fsi=np.atleast_1d(p["F_SI"]).astype(float),
                   bsi=np.atleast_1d(p["B_SI"]).astype(float),
                   osi=np.atleast_1d(p["O_SI"]).astype(float))
        by_key[(match.group(2), pos.size)] = rec
        by_date[match.group(2)].append(rec)
    return by_key, by_date


def goodunit_positions(gus) -> np.ndarray:
    """Probe y-position per unit, dereferencing the HDF5 reference layout if present."""
    sp = np.array(gus["spikepos"])
    if sp.dtype == h5py.ref_dtype:
        return np.array([np.array(gus.file[sp[i][0]]).squeeze()[1] for i in range(len(sp))], float)
    return (sp[1] if sp.shape[0] == 2 else sp[:, 1]).astype(float)


def goodunit_unittypes(f, gus, n_units: int) -> np.ndarray:
    if gus["unittype"].dtype == h5py.ref_dtype:
        return np.array([np.array(f[gus["unittype"][u][0]]).squeeze() for u in range(n_units)]).astype(int)
    return np.array([np.array(gus["unittype"][u]).squeeze() for u in range(n_units)]).astype(int)


def match_session(by_key: dict, by_date: dict, date: str, n_units: int, pos_gu: np.ndarray):
    """Return the Processed record whose units align with this GoodUnit file, or None."""
    rec = by_key.get((date, n_units))
    if rec is not None and np.allclose(np.sort(rec["pos"]), np.sort(pos_gu), atol=0.5):
        if np.allclose(rec["pos"], pos_gu, atol=0.5):
            return rec
    for candidate in by_date.get(date, []):
        if candidate["pos"].size == n_units and np.allclose(candidate["pos"], pos_gu, atol=0.5):
            return candidate
    return None


def main():
    args = parse_args()
    bins = [(t, t + args.bin) for t in range(args.tstart, args.tend, args.bin)]
    area = pd.read_excel(f"{args.raw_root}/exclude_area.xls", engine="xlrd")
    area.columns = [c.strip() for c in area.columns]
    domain = session_domain(area)
    by_key, by_date = index_processed(f"{args.raw_root}/Processed")

    stim = pd.read_csv(args.stim_csv).sort_values("tn_index")
    gu_files = sorted(glob.glob(f"{args.raw_root}/GoodUnit/GoodUnit_*.mat"))
    if args.limit:
        gu_files = gu_files[:args.limit]
    print(f"sessions: {len(gu_files)}  bins: {len(bins)} ({args.tstart}-{args.tend}ms @ {args.bin}ms)")

    blocks, cols = [], {k: [] for k in ("neuroid_id", "region", "animal", "reliability",
                                        "pos", "unittype", "F_SI", "B_SI", "O_SI")}
    matched, skipped = 0, []
    for gf in gu_files:
        name = os.path.basename(gf)
        try:
            date = re.match(r"GoodUnit_(\d+)_", name).group(1)
            with h5py.File(gf, "r") as f:
                gus = f["GoodUnitStrc"]
                pre = int(np.array(f["global_params"]["pre_onset"]).squeeze())
                n_units = len(gus["response_matrix_img"])
                pos_gu = goodunit_positions(gus)
                ut_gu = goodunit_unittypes(f, gus, n_units)
                rec = match_session(by_key, by_date, date, n_units, pos_gu)
                if rec is None:
                    print(f"  SKIP {name}: no Processed match (nU={n_units})")
                    skipped.append(name)
                    continue
                ses, animal = rec["ses"], rec["animal"]
                session_psth = np.zeros((1000, len(bins), n_units), np.float32)
                for u in range(n_units):
                    psth = np.array(f[gus["response_matrix_img"][u][0]])[:, :1000]  # time x img
                    for bi, (s, e) in enumerate(bins):
                        session_psth[:, bi, u] = psth[pre + s: pre + e].mean(axis=0)
            for u in range(n_units):
                blocks.append(session_psth[:, :, u])
                cols["neuroid_id"].append(f"{ses}.{u}")
                cols["region"].append(region_of(area, domain, pos_gu[u], ses))
                cols["animal"].append(animal)
                cols["reliability"].append(rec["rel"][u])
                cols["pos"].append(pos_gu[u])
                cols["unittype"].append(UTYPE.get(ut_gu[u], "?"))
                cols["F_SI"].append(rec["fsi"][u])
                cols["B_SI"].append(rec["bsi"][u])
                cols["O_SI"].append(rec["osi"][u])
            matched += 1
            print(f"  ses{ses:02d} {animal} {name[:34]}: {n_units} units")
        except Exception as e:  # truncated/unreadable file -> skip the session, keep the rest
            print(f"  SKIP {name}: {type(e).__name__}: {e}")
            skipped.append(name)

    print(f"\nmatched {matched}/{len(gu_files)} sessions; skipped {len(skipped)}: {skipped}")
    data = np.transpose(np.stack(blocks, axis=-1), (0, 2, 1))  # presentation x neuroid x time_bin
    assembly = xr.DataArray(
        data, dims=("presentation", "neuroid", "time_bin"),
        coords={
            "stimulus_id": ("presentation", stim.stimulus_id.values),
            "nsd_id": ("presentation", stim.nsd_id.values),
            "neuroid_id": ("neuroid", cols["neuroid_id"]),
            "region": ("neuroid", cols["region"]),
            "animal": ("neuroid", cols["animal"]),
            "reliability": ("neuroid", np.array(cols["reliability"], np.float32)),
            "pos": ("neuroid", np.array(cols["pos"], np.float32)),
            "unittype": ("neuroid", cols["unittype"]),
            "F_SI": ("neuroid", np.array(cols["F_SI"], np.float32)),
            "B_SI": ("neuroid", np.array(cols["B_SI"], np.float32)),
            "O_SI": ("neuroid", np.array(cols["O_SI"], np.float32)),
            "time_bin_start": ("time_bin", [s for s, _ in bins]),
            "time_bin_end": ("time_bin", [e for _, e in bins]),
        })
    print(f"temporal assembly: {dict(assembly.sizes)}")
    reliable = assembly.region.values[assembly.reliability.values > 0.4]
    print("reliable (>0.4) per region:")
    print(pd.Series(reliable).value_counts().to_string())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    assembly.to_netcdf(args.out)
    print(f"saved -> {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
