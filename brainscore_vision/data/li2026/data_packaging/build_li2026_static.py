"""Build the Li2026 (Triple-N) static neural assembly and stimulus mapping from the raw
ScienceDB release. Run this before :mod:`package_li2026` (which uploads the outputs to S3).

Raw source -- Triple-N, ScienceDB doi:10.57760/sciencedb.33556::

  <raw-root>/Processed/Processed_ses*.mat   per-session trial-averaged responses:
                                            response_best (units x 1072), reliability_best,
                                            pos, UnitType (1=SU/2=MUA/3=NonSom),
                                            F_SI/B_SI/O_SI, snr, best_r_time1/2
  <raw-root>/exclude_area.xls               per-session area table: SesIdx, y1, y2,
                                            AREALABEL, Area in {IT, EVC}

NSD metadata, for aligning the 1000 presentations to Allen2022 (not part of Triple-N)::

  <nsd-meta>/nsd_expdesign.mat              sharedix: Triple-N image order -> 1-indexed NSD id
  <nsd-meta>/nsd_stim_info_merged.csv       cocoId per 0-indexed NSD id

Outputs (consumed by package_li2026.py and build_li2026_temporal.py)::

  <out-dir>/Li2026_stimulus_set.csv         stimulus_id (Allen2022-aligned nsd_<0-indexed>),
                                            nsd_id (0-indexed), nsd_id_1indexed, tn_index, coco_id
  <out-dir>/Li2026_static_assembly.nc       presentation(1000) x neuroid x time_bin(1)

Region rule: IT sessions pool every unit as ``IT`` (patch label kept in ``arealabel``);
EVC sessions assign V1/V2/V4 by probe position (``y1 < pos < y2`` -> ``AREALABEL`` prefix).
Neuroids are NOT reliability-filtered here -- the benchmark applies reliability > 0.4 at
load time, so the assembly stays a faithful copy of the source.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr

UTYPE = {1: "SU", 2: "MUA", 3: "NonSom"}
# Representative analysis window (ms post-onset) carried on the single time bin.
TIME_BIN_START, TIME_BIN_END = 70, 220


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default="/Volumes/Hagibis/triple-n",
                    help="directory containing Processed/ and exclude_area.xls")
    ap.add_argument("--nsd-meta", default="/Volumes/Hagibis/nsd/metadata",
                    help="directory containing nsd_expdesign.mat and nsd_stim_info_merged.csv")
    ap.add_argument("--out-dir", default="/Volumes/Hagibis/triple-n/build")
    return ap.parse_args()


def load_area_table(area_path: str):
    """Return the area DataFrame and a per-session domain map (IT or EVC)."""
    area = pd.read_excel(area_path, engine="xlrd")
    area.columns = [c.strip() for c in area.columns]
    domain = {ses: ("EVC" if set(grp.Area.astype(str).str.strip().str.upper()) == {"EVC"} else "IT")
              for ses, grp in area.groupby("SesIdx")}
    return area, domain


def region_label(area: pd.DataFrame, domain: dict, pos: float, ses: int):
    """Map a unit's probe position to (region, arealabel) for its session."""
    dom = domain[ses]
    for _, r in area[area.SesIdx == ses].iterrows():
        if r.y1 < pos < r.y2:
            label = str(r.AREALABEL).strip()
            if dom == "EVC":
                for v in ("V1", "V2", "V4"):
                    if label.startswith(v):
                        return v, label
                return "EVC-other", label
            return "IT", label
    return ("IT", "IT-other") if dom == "IT" else ("EVC-other", "none")


def build_stimulus_table(nsd_meta: str) -> pd.DataFrame:
    """Triple-N presentation index (1..1000) -> Allen2022-aligned NSD stimulus_id."""
    shared = sio.loadmat(f"{nsd_meta}/nsd_expdesign.mat")["sharedix"].squeeze().astype(int)  # 1-indexed
    stim_info = pd.read_csv(f"{nsd_meta}/nsd_stim_info_merged.csv")  # row order = 0-indexed NSD id
    coco_col = next((c for c in stim_info.columns if c.lower() == "cocoid"), None)
    coco = stim_info[coco_col].values if coco_col else np.full(73000, -1)
    stim = pd.DataFrame({
        "tn_index": np.arange(1, 1001),       # Triple-N presentation order
        "nsd_id": shared - 1,                 # 0-indexed, matches Allen2022
        "nsd_id_1indexed": shared,            # raw sharedix value
        "coco_id": [int(coco[s - 1]) for s in shared],
    })
    stim["stimulus_id"] = ["nsd_%05d" % n for n in stim.nsd_id]  # Allen2022 convention
    assert stim.stimulus_id.is_unique
    return stim[["stimulus_id", "nsd_id", "nsd_id_1indexed", "tn_index", "coco_id"]]


def build_assembly(processed_dir: str, area: pd.DataFrame, domain: dict,
                   stim: pd.DataFrame) -> xr.DataArray:
    cols = {k: [] for k in ("resp", "neuroid_id", "region", "animal", "arealabel", "unittype",
                            "reliability", "pos", "F_SI", "B_SI", "O_SI", "snr",
                            "best_time_start", "best_time_end")}
    for f in sorted(glob.glob(f"{processed_dir}/Processed_ses*.mat")):
        ses = int(re.search(r"ses(\d+)", os.path.basename(f)).group(1))
        monkey = re.search(r"_(M\d)_", os.path.basename(f)).group(1)
        m = sio.loadmat(f, squeeze_me=True)
        responses = np.atleast_2d(np.atleast_1d(m["response_best"]).astype(np.float32))[:, :1000]
        pos = np.atleast_1d(m["pos"]).astype(float)
        for u in range(responses.shape[0]):
            region, label = region_label(area, domain, pos[u], ses)
            cols["resp"].append(responses[u])
            cols["neuroid_id"].append(f"{ses}.{u}")
            cols["region"].append(region)
            cols["animal"].append(monkey)
            cols["arealabel"].append(label)
            cols["unittype"].append(UTYPE.get(int(np.atleast_1d(m["UnitType"])[u]), "?"))
            cols["reliability"].append(float(np.atleast_1d(m["reliability_best"])[u]))
            cols["pos"].append(pos[u])
            cols["F_SI"].append(float(np.atleast_1d(m["F_SI"])[u]))
            cols["B_SI"].append(float(np.atleast_1d(m["B_SI"])[u]))
            cols["O_SI"].append(float(np.atleast_1d(m["O_SI"])[u]))
            cols["snr"].append(float(np.atleast_1d(m["snr"])[u]))
            cols["best_time_start"].append(int(np.atleast_1d(m["best_r_time1"])[u]))
            cols["best_time_end"].append(int(np.atleast_1d(m["best_r_time2"])[u]))

    data = np.array(cols["resp"], dtype=np.float32)[:, :, None]  # neuroid x presentation x time_bin
    assembly = xr.DataArray(
        data, dims=("neuroid", "presentation", "time_bin"),
        coords={
            "neuroid_id": ("neuroid", cols["neuroid_id"]),
            "region": ("neuroid", cols["region"]),
            "animal": ("neuroid", cols["animal"]),
            "arealabel": ("neuroid", cols["arealabel"]),
            "unittype": ("neuroid", cols["unittype"]),
            "reliability": ("neuroid", np.array(cols["reliability"], np.float32)),
            "pos": ("neuroid", np.array(cols["pos"], np.float32)),
            "F_SI": ("neuroid", np.array(cols["F_SI"], np.float32)),
            "B_SI": ("neuroid", np.array(cols["B_SI"], np.float32)),
            "O_SI": ("neuroid", np.array(cols["O_SI"], np.float32)),
            "snr": ("neuroid", np.array(cols["snr"], np.float32)),
            "best_time_start": ("neuroid", np.array(cols["best_time_start"], np.int16)),
            "best_time_end": ("neuroid", np.array(cols["best_time_end"], np.int16)),
            "stimulus_id": ("presentation", stim.stimulus_id.values),
            "nsd_id": ("presentation", stim.nsd_id.values),
            "tn_index": ("presentation", stim.tn_index.values),
            "time_bin_start": ("time_bin", [TIME_BIN_START]),
            "time_bin_end": ("time_bin", [TIME_BIN_END]),
        }).transpose("presentation", "neuroid", "time_bin")
    return assembly


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    area, domain = load_area_table(f"{args.raw_root}/exclude_area.xls")

    stim = build_stimulus_table(args.nsd_meta)
    stim.to_csv(f"{args.out_dir}/Li2026_stimulus_set.csv", index=False)
    print(f"stimulus set: {stim.shape}\n{stim.head(3).to_string(index=False)}")

    assembly = build_assembly(f"{args.raw_root}/Processed", area, domain, stim)
    print(f"\nassembly: {dict(assembly.sizes)}")
    reliable = assembly.region.values[assembly.reliability.values > 0.4]
    print("reliable (>0.4) per region:")
    print(pd.Series(reliable).value_counts().to_string())

    out = f"{args.out_dir}/Li2026_static_assembly.nc"
    assembly.to_netcdf(out)
    print(f"\nsaved -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
