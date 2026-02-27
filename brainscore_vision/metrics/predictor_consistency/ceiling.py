import numpy as np
import xarray as xr
from brainscore_core.metrics import Score
from brainscore_vision.metrics import Ceiling

def _get_presentation_level(X, name: str):
    """
    Return a 1D array aligned to 'presentation' for `name`, whether it is:
      - a real coordinate on 'presentation', or
      - a level of the presentation MultiIndex.
    """
    if name in X.coords and "presentation" in X[name].dims:
        return np.asarray(X[name].values)

    if "presentation" in X.dims and "presentation" in X.indexes:
        idx = X.indexes["presentation"]
        if hasattr(idx, "names") and name in idx.names:
            return np.asarray(idx.get_level_values(name))

    raise ValueError(
        f"Could not find '{name}' as coord-on-presentation or MultiIndex level. "
        f"dims={X.dims} coords={list(X.coords)} "
        f"presentation_index_names={getattr(X.indexes.get('presentation', None), 'names', None)}"
    )

def _pearson_per_neuroid(a, b):
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return (a * b).sum(axis=0) / np.where(denom == 0, np.nan, denom)

def _spearman_brown(r):
    return (2 * r) / (1 + r)

class SplitHalfPredictorConsistency(Ceiling):
    def __init__(self, n_splits=10, seed=0, image_level="image_id", rep_level="repetition"):
        self.n_splits = n_splits
        self.seed = seed
        self.image_level = image_level
        self.rep_level = rep_level

    def __call__(self, X) -> Score:
        if "time_bin" in X.dims and X.sizes.get("time_bin", 1) == 1:
            X = X.squeeze("time_bin")
        X = X.transpose("presentation", "neuroid")

        reps = _get_presentation_level(X, self.rep_level).astype(int)
        imgs = _get_presentation_level(X, self.image_level)

        unique_imgs, inv = np.unique(imgs, return_inverse=True)
        rng = np.random.default_rng(self.seed)

        split_rs = []
        for _ in range(self.n_splits):
            A_means, B_means = [], []
            kept = 0

            for img_idx in range(len(unique_imgs)):
                idx = np.where(inv == img_idx)[0]
                if idx.size < 2:
                    continue

                rvals = reps[idx]
                uniq_r = np.unique(rvals)
                if uniq_r.size < 2:
                    continue

                perm = rng.permutation(uniq_r)
                half = uniq_r.size // 2
                A_r, B_r = perm[:half], perm[half:]
                if A_r.size == 0 or B_r.size == 0:
                    continue

                A_idx = idx[np.isin(rvals, A_r)]
                B_idx = idx[np.isin(rvals, B_r)]
                if A_idx.size == 0 or B_idx.size == 0:
                    continue

                A_means.append(X.isel(presentation=A_idx).mean("presentation").values)
                B_means.append(X.isel(presentation=B_idx).mean("presentation").values)
                kept += 1

            if kept < 10:
                split_rs.append(np.full((X.sizes["neuroid"],), np.nan))
                continue

            A = np.stack(A_means, axis=0)
            B = np.stack(B_means, axis=0)

            r = _pearson_per_neuroid(A, B)
            split_rs.append(_spearman_brown(r))

        split_rs = np.stack(split_rs, axis=0)  # (split, neuroid)
        per_split = np.nanmedian(split_rs, axis=1)
        value = np.nanmean(per_split)
        err = np.nanstd(per_split) / np.sqrt(np.sum(np.isfinite(per_split)))

        score = Score(value)
        score.attrs["raw"] = xr.DataArray(
            split_rs,
            dims=("split", "neuroid"),
            coords={"split": np.arange(self.n_splits), "neuroid": X["neuroid"].values},
        )
        score.attrs["error"] = Score(err)
        score.attrs["ceiling"] = "predictor_consistency_split_half"
        return score
