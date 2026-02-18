import numpy as np
from brainscore_core.metrics import Metric, Score

def _find_rep_dim(X):
    for d in ["repetition", "repetition_id", "trial"]:
        if d in X.dims:
            return d
    raise ValueError(f"No repetition dim found. dims={list(X.dims)}")

class SplitHalfPredictorConsistency(Metric):
    """
    Ceiling for noisy predictor X: split-half reliability across repetitions.
    Returns per-neuroid correlation, Spearman-Brown corrected, then aggregated.
    """
    def __call__(self, X) -> Score:
        rep_dim = _find_rep_dim(X)
        reps = X[rep_dim].values
        try:
            odd = reps[(reps.astype(int) % 2) == 1]
            even = reps[(reps.astype(int) % 2) == 0]
            Xa = X.sel({rep_dim: odd}).mean(rep_dim)
            Xb = X.sel({rep_dim: even}).mean(rep_dim)
        except Exception:
            n = len(reps)
            Xa = X.isel({rep_dim: slice(0, n // 2)}).mean(rep_dim)
            Xb = X.isel({rep_dim: slice(n // 2, None)}).mean(rep_dim)

        Xa = X.sel({rep_dim: odd}).mean(rep_dim)
        Xb = X.sel({rep_dim: even}).mean(rep_dim)

        Xa = Xa.sortby("presentation")
        Xb = Xb.sortby("presentation")

        Xa = Xa.transpose("presentation", "neuroid")
        Xb = Xb.transpose("presentation", "neuroid")
        assert np.array_equal(Xa["presentation"].values, Xb["presentation"].values)

        a = Xa.values
        b = Xb.values
        a = a - a.mean(axis=0, keepdims=True)
        b = b - b.mean(axis=0, keepdims=True)
        denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
        r = (a * b).sum(axis=0) / np.where(denom == 0, np.nan, denom)

        # Spearman–Brown correction
        r_sb = (2 * r) / (1 + r)

        score = Score(r_sb, coords={"neuroid": Xa["neuroid"].values}, dims=["neuroid"])
        score = score.median("neuroid") 
        score.attrs["ceiling"] = "predictor_consistency_split_half"
        return score
