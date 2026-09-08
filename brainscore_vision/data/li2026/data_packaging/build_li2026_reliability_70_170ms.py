"""Recompute per-unit split-half (Spearman-Brown) reliability at the FIXED 70-170 ms
window from raw GoodUnit rasters -- the window-matched noise ceiling for the
fixed-window static Li2026 assembly. Streams one session at a time (low RAM).

Output: CSV neuroid_id,reliability_window  (neuroid_id = f"{ses}.{u}", GoodUnit order).
"""
import glob, os, re, sys
from collections import defaultdict
import numpy as np, pandas as pd, scipy.io as sio, h5py

GU_DIR = "/Volumes/Hagibis/triple-n/GoodUnit"
PROC_DIR = "/Volumes/Hagibis/triple-n/Processed"
OUT = "/Volumes/Hagibis/triple-n/build/Li2026_reliability_70_170ms.csv"
W0, W1, K, SEED = 70, 170, 20, 0


def goodunit_positions(gus):
    sp = np.array(gus["spikepos"])
    if sp.dtype == h5py.ref_dtype:
        return np.array([np.array(gus.file[sp[i][0]]).squeeze()[1] for i in range(len(sp))], float)
    return (sp[1] if sp.shape[0] == 2 else sp[:, 1]).astype(float)


def index_processed():
    by_key, by_date = {}, defaultdict(list)
    for pf in glob.glob(f"{PROC_DIR}/Processed_ses*.mat"):
        m = re.match(r"Processed_ses(\d+)_(\d+)_M(\d)_(\d+)\.mat", os.path.basename(pf))
        p = sio.loadmat(pf, squeeze_me=True)
        pos = np.atleast_1d(p["pos"]).astype(float)
        rec = dict(ses=int(m.group(1)), date=m.group(2), pos=pos)
        by_key[(m.group(2), pos.size)] = rec
        by_date[m.group(2)].append(rec)
    return by_key, by_date


def match_session(by_key, by_date, date, n_units, pos_gu):
    rec = by_key.get((date, n_units))
    if rec is not None and np.allclose(rec["pos"], pos_gu, atol=0.5):
        return rec
    for c in by_date.get(date, []):
        if c["pos"].size == n_units and np.allclose(c["pos"], pos_gu, atol=0.5):
            return c
    return None


def session_split_plan(img_code, n_img, rng):
    """Per split: assign each (valid) trial to half 0/1 within its image. Reused across units."""
    per_img = [np.where(img_code == j)[0] for j in range(n_img)]
    plans = []
    for _ in range(K):
        h0 = np.zeros(img_code.size, bool); h1 = np.zeros(img_code.size, bool)
        for idx in per_img:
            if idx.size < 2:
                continue
            idx = idx.copy(); rng.shuffle(idx); half = idx.size // 2
            h0[idx[:half]] = True; h1[idx[half:2 * half]] = True
        plans.append((h0, h1))
    return plans


def unit_reliability(rate, img_code, n_img, plans):
    rs = []
    for h0, h1 in plans:
        s0 = np.bincount(img_code[h0], weights=rate[h0], minlength=n_img)
        c0 = np.bincount(img_code[h0], minlength=n_img)
        s1 = np.bincount(img_code[h1], weights=rate[h1], minlength=n_img)
        c1 = np.bincount(img_code[h1], minlength=n_img)
        m = (c0 > 0) & (c1 > 0)
        if m.sum() <= 10:
            continue
        a, b = s0[m] / c0[m], s1[m] / c1[m]
        if a.std() == 0 or b.std() == 0:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if np.isfinite(r) and (1 + r) != 0:
            rs.append(2 * r / (1 + r))
    return float(np.nanmean(rs)) if rs else np.nan


def main():
    by_key, by_date = index_processed()
    gu_files = sorted(glob.glob(f"{GU_DIR}/GoodUnit_*.mat"))
    rows, skipped = [], []
    for gi, gf in enumerate(gu_files):
        name = os.path.basename(gf)
        try:
            date = re.match(r"GoodUnit_(\d+)_", name).group(1)
            with h5py.File(gf, "r") as f:
                gus = f["GoodUnitStrc"]; pre = int(np.array(f["global_params"]["pre_onset"]).squeeze())
                nU = len(gus["Raster"])
                pos_gu = goodunit_positions(gus)
                rec = match_session(by_key, by_date, date, nU, pos_gu)
                if rec is None:
                    skipped.append(name); print(f"  SKIP {name}: no match", flush=True); continue
                ses = rec["ses"]
                tvi = np.array(f["meta_data"]["trial_valid_idx"]).reshape(-1)
                dvi = np.array(f["meta_data"]["dataset_valid_idx"]).reshape(-1).astype(bool)
                timg = tvi[dvi].astype(int)
                nsd = (timg >= 1) & (timg <= 1000)
                imgs = np.unique(timg[nsd]); code_map = {im: j for j, im in enumerate(imgs)}
                img_code = np.array([code_map[i] for i in timg[nsd]])
                n_img = imgs.size
                rng = np.random.RandomState(SEED + ses)
                plans = session_split_plan(img_code, n_img, rng)
                for u in range(nU):
                    ra = np.array(f[gus["Raster"][u][0]])
                    rate = (ra[pre + W0:pre + W1].mean(0).astype(np.float32) * 1000.0)[nsd]
                    rows.append((f"{ses}.{u}", unit_reliability(rate, img_code, n_img, plans)))
            print(f"  [{gi+1}/{len(gu_files)}] ses{ses:02d} {name[:32]}: {nU} units", flush=True)
        except Exception as e:
            skipped.append(name); print(f"  SKIP {name}: {type(e).__name__}: {e}", flush=True)
    df = pd.DataFrame(rows, columns=["neuroid_id", "reliability_window"])
    df.to_csv(OUT, index=False)
    print(f"\nDONE: {len(df)} units -> {OUT}; skipped {len(skipped)}: {skipped}", flush=True)
    print(f"median reliability_window={df.reliability_window.median():.3f}; "
          f">0.4 = {int((df.reliability_window > 0.4).sum())}", flush=True)


if __name__ == "__main__":
    main()
