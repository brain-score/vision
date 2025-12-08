# -*- coding: utf-8 -*-
# PyTorch GPU port of sklearn-like _RidgeGCV (GCV/LOO ridge) with identical logic
# to the provided NumPy version. Dense CUDA supported; sparse supported with careful
# conversions only where unavoidable to preserve exact behavior.
# for reference, see https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py


from typing import Optional, Tuple, Union, Sequence
import numpy as np
import torch

import logging
_logger = logging.getLogger(__name__)

TensorLike = Union[torch.Tensor, np.ndarray]

def _as_torch(x: TensorLike, device=None, dtype=torch.float64) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if device is not None and t.device != device:
            t = t.to(device)
        return t
    t = torch.from_numpy(np.asarray(x)).to(dtype)
    if device is not None:
        t = t.to(device)
    return t

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def _is_sparse(x: torch.Tensor) -> bool:
    return x.is_sparse or x.is_sparse_csr or x.is_sparse_coo if hasattr(x, "is_sparse_coo") else (x.layout in (torch.sparse_coo, torch.sparse_csr))

def _safe_sparse_dot(A: torch.Tensor, B: torch.Tensor, *, dense_output: bool = True) -> torch.Tensor:
    # Mirrors sklearn.utils.extmath.safe_sparse_dot semantics for our use cases.
    if _is_sparse(A) and _is_sparse(B):
        out = torch.sparse.mm(A, B)  # sparse@sparse -> sparse
        return out.to_dense() if dense_output else out
    elif _is_sparse(A):
        out = torch.sparse.mm(A, B)  # sparse @ dense -> dense
        return out
    elif _is_sparse(B):
        out = torch.matmul(A, B.to_dense())  # dense @ sparse -> dense
        return out
    else:
        return torch.matmul(A, B)

def _mean_variance_axis_sparse_weighted(X: torch.Tensor, sqrt_sw: torch.Tensor, axis: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    # Weighted mean/var along axis for sparse X AFTER multiplying rows by sqrt_sw (as in sklearn)
    # We never return variance but keep signature parity; variance not used by caller here.
    if axis != 0:
        raise NotImplementedError("Only axis=0 is used in this port.")
    # scale rows, then compute mean columnwise: mean = sum_i w_i * x_i / sum_i w_i with w_i = 1 (after premultiplying by sqrt_sw)
    # In sklearn path we pass X_weighted already multiplied by sqrt_sw on rows.
    if _is_sparse(X):
        Xw = torch.sparse.mm(torch.diag(sqrt_sw), X.to_sparse_csr() if not X.is_sparse else X)  # [n,n] x [n,p] -> [n,p]
        Xw_d = Xw.to_dense()
    else:
        Xw_d = sqrt_sw[:, None] * X
    weight_sum = torch.dot(sqrt_sw, sqrt_sw)  # sum of original sample weights
    mean = Xw_d.sum(dim=0) / weight_sum
    var = torch.zeros_like(mean)
    return mean, var

def _find_smallest_angle(v: torch.Tensor, Q: torch.Tensor) -> int:
    # Find column of Q most aligned to v (max absolute cosine); Q expected orthonormal columns.
    # Works for both Q in Gram eigvecs and U in SVD branch.
    v = v / (torch.linalg.norm(v) + 1e-18)
    proj = torch.abs(Q.T @ v)  # [k]
    return int(torch.argmax(proj).item())

def _check_sample_weight(sample_weight: Union[float, TensorLike, None], X: torch.Tensor, dtype) -> torch.Tensor:
    n = X.shape[0]
    if sample_weight is None:
        return torch.ones(n, dtype=dtype, device=X.device)
    if isinstance(sample_weight, (float, int)):
        return torch.full((n,), float(sample_weight), dtype=dtype, device=X.device)
    sw = _as_torch(sample_weight, device=X.device, dtype=dtype).flatten()
    if sw.numel() != n:
        raise ValueError("sample_weight must have shape (n_samples,)")
    return sw

def _preprocess_data(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    fit_intercept: bool,
    copy: bool,
    sample_weight: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mirrors sklearn._preprocess_data used by linear models (without standardize).
    Dense X: center X and y if fit_intercept True; sparse X: DO NOT center here.
    Returns (X_proc, y_proc, X_offset, y_offset, X_scale) with X_scale=1.
    """
    if copy:
        X = X.clone()
        y = y.clone()

    n_samples, n_features = X.shape[0], X.shape[1]
    X_scale = torch.ones(n_features, dtype=X.dtype, device=X.device)
    if not fit_intercept:
        X_offset = torch.zeros(n_features, dtype=X.dtype, device=X.device)
        if y.ndim == 1:
            y_offset = torch.tensor(0, dtype=X.dtype, device=X.device)
        else:
            y_offset = torch.zeros(y.shape[1], dtype=X.dtype, device=X.device)
        return X, y, X_offset, y_offset, X_scale

    # fit_intercept == True
    if _is_sparse(X):
        # sparse path: no centering here (exactly as sklearn). Offsets are zeros.
        X_offset = torch.zeros(n_features, dtype=X.dtype, device=X.device)
        if y.ndim == 1:
            # weighted mean of y (will be added back after predictions)
            if sample_weight is None:
                y_offset = y.mean(dim=0)
            else:
                sw = sample_weight
                y_offset = (sw @ y) / (sw.sum() + 1e-18)
            y = y - y_offset
        else:
            if sample_weight is None:
                y_offset = y.mean(dim=0)
            else:
                sw = sample_weight[:, None]
                y_offset = (sw * y).sum(dim=0) / (sample_weight.sum() + 1e-18)
            y = y - y_offset
        return X, y, X_offset, y_offset, X_scale

    # dense path: center X and y
    if sample_weight is None:
        X_offset = X.mean(dim=0)
    else:
        sw = sample_weight[:, None]  # [n,1]
        X_offset = (sw * X).sum(dim=0) / (sample_weight.sum() + 1e-18)
    X = X - X_offset

    if y.ndim == 1:
        if sample_weight is None:
            y_offset = y.mean(dim=0)
        else:
            y_offset = (sample_weight @ y) / (sample_weight.sum() + 1e-18)
        y = y - y_offset
    else:
        if sample_weight is None:
            y_offset = y.mean(dim=0)
        else:
            sw = sample_weight[:, None]
            y_offset = (sw * y).sum(dim=0) / (sample_weight.sum() + 1e-18)
        y = y - y_offset

    return X, y, X_offset, y_offset, X_scale

def _rescale_data(X: torch.Tensor, y: torch.Tensor, sample_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Multiply rows of X and y by sqrt(sample_weight) to turn SWLS into unweighted LS (as in sklearn)
    sqrt_sw = torch.sqrt(sample_weight.clamp_min(0))
    if _is_sparse(X):
        Xw = torch.sparse.mm(torch.diag(sqrt_sw), X.to_sparse_csr() if not X.is_sparse else X)
    else:
        Xw = sqrt_sw[:, None] * X
    if y.ndim == 1:
        yw = sqrt_sw * y
    else:
        yw = sqrt_sw[:, None] * y
    return Xw, yw, sqrt_sw

def _check_gcv_mode(X: torch.Tensor, gcv_mode: Optional[str]) -> str:
    if gcv_mode in ("eigen", "svd"):
        return gcv_mode
    if _is_sparse(X):
        return "svd"  # matches sklearn's choice for sparse
    # dense: eigen if n_samples <= n_features, else svd
    return "eigen" if X.shape[0] <= X.shape[1] else "svd"

def _X_CenterStack_dense(X: torch.Tensor, X_mean: torch.Tensor, sqrt_sw: torch.Tensor) -> torch.Tensor:
    # (X - X_mean) with sample-weight centering proxy used in covariance+intercept branch,
    # then append intercept column sqrt_sw
    return torch.cat([X - (sqrt_sw[:, None] * 0 + 1) * X_mean, sqrt_sw[:, None]], dim=1)

def _X_CenterStack_sparse(X: torch.Tensor, X_mean: torch.Tensor, sqrt_sw: torch.Tensor) -> torch.Tensor:
    # Build dense representation of centered+intercept matrix as in sklearn's _X_CenterStackOp
    Xd = X.to_dense() if _is_sparse(X) else X
    Xc = Xd - (sqrt_sw[:, None] * 0 + 1) * X_mean
    return torch.cat([Xc, sqrt_sw[:, None]], dim=1)

class RidgeGCVTorch:
    """
    Torch/CUDA port of your NumPy/_RidgeGCV.
    API mirrors the essentials: fit(...), attributes: alpha_, best_score_, dual_coef_, coef_, intercept_, cv_results_ (optional).
    All math done in float64 by default to match NumPy.
    """

    def __init__(
        self,
        alphas: Union[Sequence[float], float] = (0.1, 1.0, 10.0),
        *,
        fit_intercept: bool = True,
        scoring=None,
        copy_X: bool = True,
        gcv_mode: Optional[str] = None,
        store_cv_results: bool = False,
        is_clf: bool = False,
        alpha_per_target: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
        pbar: bool = False,
        store_results_gpu: bool = True, #adds ability to not store cv_results_ on GPU to save VRAM (originally on GPU)
    ):
        self.alphas = np.asarray(alphas) if not np.isscalar(alphas) else np.array(alphas)
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.copy_X = copy_X
        self.gcv_mode = gcv_mode
        self.store_cv_results = store_cv_results
        self.is_clf = is_clf
        self.alpha_per_target = alpha_per_target
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.pbar = pbar
        self.store_results_gpu = store_results_gpu

        # learned attributes
        self.alpha_ = None
        self.best_score_ = None
        self.dual_coef_ = None
        self.coef_ = None
        self.intercept_ = None
        self.cv_results_ = None

    # ---- helpers mirror the NumPy version ----
    @staticmethod
    def _decomp_diag(v_prime: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        # diag( Q @ diag(v_prime) @ Q^T ) = sum_k v_k * Q[:,k]^2
        return (v_prime * (Q * Q)).sum(dim=-1)

    @staticmethod
    def _diag_dot(D: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if B.ndim > 1:
            shape = (slice(None),) + (None,) * (B.ndim - 1)
            D = D[shape]
        return D * B

    def _compute_gram(self, X: torch.Tensor, sqrt_sw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dense: X already centered in preprocessing OR no intercept -> just X X^T
        # Sparse: emulate weighted-centering trick exactly as sklearn
        if not _is_sparse(X) or not self.fit_intercept:
            X_mean = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
            gram = _safe_sparse_dot(X, X.T, dense_output=True)
            return gram, X_mean

        # sparse + fit_intercept
        n_samples = X.shape[0]
        # Weighted mean after row scaling by sqrt_sw (sklearn path)
        X_weighted = torch.sparse.mm(torch.diag(sqrt_sw), X.to_sparse_csr() if not X.is_sparse else X)
        X_mean, _ = _mean_variance_axis_sparse_weighted(X_weighted, sqrt_sw, axis=0)
        X_mean = X_mean * n_samples / (sqrt_sw @ sqrt_sw)
        # Terms: X_mX = sqrt_sw[:,None] * (X_mean @ X^T),  X_mX_m = (sqrt_sw sqrt_sw^T) * (X_mean·X_mean)
        Xt = _safe_sparse_dot(X, X.T, dense_output=True)
        X_mean_row = _safe_sparse_dot(X_mean[None, :], X.T, dense_output=True)  # [1,n]
        X_mX = sqrt_sw[:, None] * X_mean_row  # [n, n]
        X_mX_m = torch.outer(sqrt_sw, sqrt_sw) * (X_mean @ X_mean)
        gram = Xt + X_mX_m - X_mX - X_mX.T
        return gram, X_mean

    def _compute_covariance(self, X: torch.Tensor, sqrt_sw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Only called for sparse X in sklearn logic
        if not self.fit_intercept:
            X_mean = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
            cov = _safe_sparse_dot(X.T, X, dense_output=True)
            return cov, X_mean

        n_samples = X.shape[0]
        X_weighted = torch.sparse.mm(torch.diag(sqrt_sw), X.to_sparse_csr() if not X.is_sparse else X)
        X_mean, _ = _mean_variance_axis_sparse_weighted(X_weighted, sqrt_sw, axis=0)
        X_mean = X_mean * n_samples / (sqrt_sw @ sqrt_sw)
        weight_sum = sqrt_sw @ sqrt_sw
        cov = _safe_sparse_dot(X.T, X, dense_output=True) - weight_sum * torch.outer(X_mean, X_mean)
        return cov, X_mean

    def _sparse_multidot_diag(self, X: torch.Tensor, A: torch.Tensor, X_mean: torch.Tensor, sqrt_sw: torch.Tensor) -> torch.Tensor:
        # Compute diag( (X - X_mean).A.(X - X_mean)^T ) without forming centered X; batch to conserve memory.
        n_samples, n_features = X.shape
        intercept_col = sqrt_sw
        scale = sqrt_sw
        batch_size = n_features  # mirror original choice
        diag = torch.empty(n_samples, dtype=X.dtype, device=X.device)
        # We need dense rows to multiply by A; we mirror numpy path X[batch].toarray()
        for start in range(0, n_samples, batch_size):
            end = min(n_samples, start + batch_size)
            idx = slice(start, end)
            Xb_dense = X[idx].to_dense() if _is_sparse(X) else X[idx]
            if self.fit_intercept:
                X_batch = torch.empty((Xb_dense.shape[0], n_features + 1), dtype=X.dtype, device=X.device)
                X_batch[:, :-1] = Xb_dense - X_mean * scale[idx][:, None]
                X_batch[:, -1] = intercept_col[idx]
            else:
                X_batch = Xb_dense
            AX = X_batch @ A
            diag[idx] = (AX * X_batch).sum(dim=1)
        return diag

    # ----- Decompose/Solve branches -----

    def _eigen_decompose_gram(self, X: torch.Tensor, y: torch.Tensor, sqrt_sw: torch.Tensor):
        K, X_mean = self._compute_gram(X, sqrt_sw)
        if self.fit_intercept:
            K = K + torch.outer(sqrt_sw, sqrt_sw)
        eigvals, Q = torch.linalg.eigh(K)  # ascending
        QT_y = Q.T @ y
        return X_mean, eigvals, Q, QT_y

    def _solve_eigen_gram(self, alpha, y, sqrt_sw, X_mean, eigvals, Q, QT_y):
        w = 1.0 / (eigvals + alpha)
        if self.fit_intercept:
            normalized_sw = sqrt_sw / (torch.linalg.norm(sqrt_sw) + 1e-18)
            intercept_dim = _find_smallest_angle(normalized_sw, Q)
            w[intercept_dim] = 0.0  # cancel regularization
        c = Q @ self._diag_dot(w, QT_y)
        Ginv_diag = self._decomp_diag(w, Q)
        if y.ndim != 1:
            Ginv_diag = Ginv_diag[:, None]
        return Ginv_diag, c

    def _eigen_decompose_covariance(self, X: torch.Tensor, y: torch.Tensor, sqrt_sw: torch.Tensor):
        n_samples, n_features = X.shape
        cov = torch.empty((n_features + 1, n_features + 1), dtype=X.dtype, device=X.device)
        cov[:-1, :-1], X_mean = self._compute_covariance(X, sqrt_sw)
        if not self.fit_intercept:
            cov = cov[:-1, :-1]
        else:
            cov[-1, :] = 0
            cov[:, -1] = 0
            cov[-1, -1] = (sqrt_sw @ sqrt_sw)
        nullspace_dim = max(0, n_features - n_samples)
        eigvals, V = torch.linalg.eigh(cov)
        eigvals = eigvals[nullspace_dim:]
        V = V[:, nullspace_dim:]
        return X_mean, eigvals, V, X

    def _solve_eigen_covariance_no_intercept(self, alpha, y, sqrt_sw, X_mean, eigvals, V, X):
        w = 1.0 / (eigvals + alpha)
        A = (V * w) @ V.T
        AXy = A @ _safe_sparse_dot(X.T, y, dense_output=True)
        y_hat = _safe_sparse_dot(X, AXy, dense_output=True)
        hat_diag = self._sparse_multidot_diag(X, A, X_mean, sqrt_sw)
        if y.ndim != 1:
            hat_diag = hat_diag[:, None]
        return (1.0 - hat_diag) / alpha, (y - y_hat) / alpha

    def _solve_eigen_covariance_intercept(self, alpha, y, sqrt_sw, X_mean, eigvals, V, X):
        intercept_sv = torch.zeros(V.shape[0], dtype=V.dtype, device=V.device)
        intercept_sv[-1] = 1.0
        intercept_dim = _find_smallest_angle(intercept_sv, V)
        w = 1.0 / (eigvals + alpha)
        w[intercept_dim] = 1.0 / eigvals[intercept_dim]
        A = (V * w) @ V.T
        # Add column sqrt_sw, and center X by the weighted trick
        X_op = _X_CenterStack_sparse(X, X_mean, sqrt_sw)
        AXy = A @ (X_op.T @ y)
        y_hat = X_op @ AXy
        hat_diag = self._sparse_multidot_diag(X, A, X_mean, sqrt_sw)
        if y.ndim != 1:
            hat_diag = hat_diag[:, None]
        return (1.0 - hat_diag) / alpha, (y - y_hat) / alpha

    def _solve_eigen_covariance(self, alpha, y, sqrt_sw, X_mean, eigvals, V, X):
        if self.fit_intercept:
            return self._solve_eigen_covariance_intercept(alpha, y, sqrt_sw, X_mean, eigvals, V, X)
        return self._solve_eigen_covariance_no_intercept(alpha, y, sqrt_sw, X_mean, eigvals, V, X)

    def _svd_decompose_design_matrix(self, X: torch.Tensor, y: torch.Tensor, sqrt_sw: torch.Tensor):
        X_mean = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
        X_aug = X
        if self.fit_intercept:
            X_aug = torch.cat([X, sqrt_sw[:, None]], dim=1)
        U, S, Vh = torch.linalg.svd(X_aug, full_matrices=False)
        singvals_sq = S * S
        UT_y = U.T @ y
        return X_mean, singvals_sq, U, UT_y

    def _solve_svd_design_matrix(self, alpha, y, sqrt_sw, X_mean, singvals_sq, U, UT_y):
        w = (1.0 / (singvals_sq + alpha)) - (1.0 / alpha)
        if self.fit_intercept:
            normalized_sw = sqrt_sw / (torch.linalg.norm(sqrt_sw) + 1e-18)
            intercept_dim = _find_smallest_angle(normalized_sw, U)
            w[intercept_dim] = -(1.0 / alpha)
        c = U @ self._diag_dot(w, UT_y) + (1.0 / alpha) * y
        Ginv_diag = self._decomp_diag(w, U) + (1.0 / alpha)
        if y.ndim != 1:
            Ginv_diag = Ginv_diag[:, None]
        return Ginv_diag, c

    # ---- public API ----

    def fit(
        self,
        X_in: TensorLike,
        y_in: TensorLike,
        sample_weight: Optional[Union[float, TensorLike]] = None,
        score_params: Optional[dict] = None,
    ):
        # Ensure torch, float64, device
        X = _as_torch(X_in, device=self.device, dtype=self.dtype)
        y = _as_torch(y_in, device=self.device, dtype=self.dtype)

        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim == 1:
            y = y
        elif y.ndim == 2 and y.shape[0] == X.shape[0]:
            pass
        else:
            raise ValueError("y must be shape (n_samples,) or (n_samples, n_targets)")

        _logger.info(f"X has shape {X.shape}, y has shape {y.shape}")

        # Handle sample_weight
        sw = None if sample_weight is None else _check_sample_weight(sample_weight, X, X.dtype)

        # Cache original y for external scoring path
        unscaled_y = y.clone()

        # Preprocess (dense: center; sparse: only y centered if fit_intercept)
        Xp, yp, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, copy=self.copy_X, sample_weight=sw
        )
        _logger.info("Data preprocessing complete")

        # Decide gcv mode (auto/eigen/svd)
        gcv_mode = _check_gcv_mode(Xp, self.gcv_mode)
        _logger.info(f"gcv mode is {gcv_mode}")
        # Choose decomposition/solve routines
        if gcv_mode == "eigen":
            decompose = self._eigen_decompose_gram
            solve = self._solve_eigen_gram
        else:  # "svd"
            if _is_sparse(Xp):
                decompose = self._eigen_decompose_covariance
                solve = self._solve_eigen_covariance
            else:
                decompose = self._svd_decompose_design_matrix
                solve = self._solve_svd_design_matrix

        n_samples = Xp.shape[0]

        # Rescale data for SWLS
        if sw is not None:
            Xp, yp, sqrt_sw = _rescale_data(Xp, yp, sw)
        else:
            sqrt_sw = torch.ones(n_samples, dtype=Xp.dtype, device=Xp.device)

        # Decompose once
        X_mean, *decomp = decompose(Xp, yp, sqrt_sw)

        n_y = 1 if yp.ndim == 1 else yp.shape[1]
        alphas_vec = torch.atleast_1d(_as_torch(self.alphas, device=Xp.device, dtype=Xp.dtype))
        n_alphas = int(alphas_vec.numel())

        if self.store_cv_results:
            if self.store_results_gpu:
                self.cv_results_ = torch.empty((n_samples * n_y, n_alphas), dtype=Xp.dtype, device=Xp.device)
            else:
                self.cv_results_ = torch.empty((n_samples * n_y, n_alphas), dtype=Xp.dtype, device="cpu")

        best_coef = None
        best_score = None
        best_alpha = None
        
        if self.pbar:
            from tqdm.auto import tqdm
            alpha_iter = tqdm(enumerate(alphas_vec), desc="Alpha loop", total=n_alphas)
        else:
            alpha_iter = enumerate(alphas_vec)

        _logger.info("Iterating alphas")
        for i, alpha in alpha_iter:
            Ginv_diag, c = solve(float(alpha.item()), yp, sqrt_sw, X_mean, *decomp)

            if self.scoring is None:
                squared_errors = (c / Ginv_diag) ** 2
                if self.alpha_per_target and n_y > 1:
                    alpha_score = -squared_errors.mean(dim=0)
                else:
                    alpha_score = -squared_errors.mean()
                if self.store_cv_results:
                    if self.store_results_gpu:
                        self.cv_results_[:, i] = squared_errors.reshape(-1)
                    else:
                        self.cv_results_[:, i] = squared_errors.reshape(-1).cpu()
            else:
                predictions = yp - (c / Ginv_diag)
                if sw is not None:
                    predictions = predictions / (sqrt_sw[:, None] if predictions.ndim > 1 else sqrt_sw)
                predictions = predictions + y_offset  # back to original scale

                if self.store_cv_results:
                    if self.store_results_gpu:
                        self.cv_results_[:, i] = predictions.reshape(-1)
                    else:
                        self.cv_results_[:, i] = predictions.reshape(-1).cpu()

                score_params = score_params or {}
                # External scorer may be a sklearn scorer expecting numpy
                pred_np = _to_numpy(predictions)
                y_np = _to_numpy(unscaled_y)
                if self.is_clf:
                    # emulate IdentityClassifier on argmax targets
                    target = y_np.argmax(axis=1)
                    alpha_score = self.scoring(None, pred_np, target, **score_params)
                else:
                    if self.alpha_per_target and n_y > 1:
                        scores = []
                        for j in range(n_y):
                            scores.append(self.scoring(None, pred_np[:, j], y_np[:, j], **score_params))
                        alpha_score = torch.from_numpy(np.array(scores)).to(self.dtype).to(self.device)
                    else:
                        alpha_score = self.scoring(None, pred_np, y_np, **score_params)
                        if not isinstance(alpha_score, torch.Tensor):
                            alpha_score = torch.tensor(alpha_score, dtype=self.dtype, device=self.device)

            # Track best (handle per-target)
            if best_score is None:
                if self.alpha_per_target and n_y > 1:
                    best_coef = c
                    best_score = torch.atleast_1d(alpha_score)
                    best_alpha = torch.full((n_y,), alpha.item(), dtype=self.dtype, device=self.device)
                else:
                    best_coef = c
                    best_score = alpha_score
                    best_alpha = torch.tensor(alpha.item(), dtype=self.dtype, device=self.device)
            else:
                if self.alpha_per_target and n_y > 1:
                    to_update = alpha_score > best_score
                    if to_update.any():
                        if c.ndim == 1:
                            best_coef[to_update] = c[to_update]
                        else:
                            best_coef[:, to_update] = c[:, to_update]
                        best_score[to_update] = alpha_score[to_update]
                        best_alpha[to_update] = alpha.item()
                else:
                    if alpha_score > best_score:
                        best_coef, best_score, best_alpha = c, alpha_score, torch.tensor(alpha.item(), dtype=self.dtype, device=self.device)

        self.alpha_ = _to_numpy(best_alpha) if isinstance(best_alpha, torch.Tensor) else best_alpha
        _logger.info(f"Best alpha selected: {self.alpha}")

        self.best_score_ = _to_numpy(best_score) if isinstance(best_score, torch.Tensor) else best_score
        self.dual_coef_ = best_coef  # shape [n_samples] or [n_samples, n_targets]

        # Primal coefficients: w = X^T c  (using preprocessed Xp)
        coef = _safe_sparse_dot(self.dual_coef_.T, Xp, dense_output=True)  # [T, p]
        if yp.ndim == 1 or yp.shape[1] == 1:
            coef = coef.ravel()
        self.coef_ = coef

        # Adjust offsets as in sklearn
        if _is_sparse(Xp):
            X_offset_eff = X_mean * X_scale  # X_scale is ones
        else:
            X_offset_eff = X_offset + X_mean * X_scale

        # Set intercept and unscale coef (X_scale==1 here but keep identical form)
        if self.coef_.ndim == 1:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = (y_offset - (X_offset_eff @ self.coef_))
            self.intercept_ = self.intercept_.item() if isinstance(self.intercept_, torch.Tensor) and self.intercept_.numel()==1 else self.intercept_
        else:
            self.coef_ = self.coef_ / X_scale[None, :]
            self.intercept_ = y_offset - (self.coef_ @ X_offset_eff)  # [T]
            # keep as 1D tensor/np array

        if self.store_cv_results:
            if yp.ndim == 1:
                shape = (n_samples, n_alphas)
            else:
                shape = (n_samples, n_y, n_alphas)
            self.cv_results_ = self.cv_results_.reshape(shape)

        self.to_cpu()

        return self

    def predict(self, X: TensorLike) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict().")

        self.to_device(self.device, predict_only=True)

        X = _as_torch(X, device=self.device, dtype=self.dtype)
        if self.coef_.ndim == 1:
            yhat = X @ self.coef_
            yhat = yhat + (self.intercept_ if isinstance(self.intercept_, torch.Tensor) else torch.tensor(self.intercept_, dtype=X.dtype, device=X.device))
        else:
            yhat = X @ self.coef_.T
            bias = self.intercept_
            if not isinstance(bias, torch.Tensor):
                bias = _as_torch(bias, device=X.device, dtype=X.dtype)
            yhat = yhat + bias
        self.to_device("cpu", predict_only=True)
        return _to_numpy(yhat)

    def to_device(self, device, predict_only: bool = False):
        """
        Move parameters to specified device.
        Without this, coef_, dual_coef_, intercept_, cv_results_ remain on the GPU after fitting.
        If predict_only, move only vars needed for prediction.
        """
        if isinstance(self.coef_, torch.Tensor):
            self.coef_ = self.coef_.to(device)
        if isinstance(self.intercept_, torch.Tensor):
            self.intercept_ = self.intercept_.to(device)
        
        if not predict_only:
            if isinstance(self.dual_coef_, torch.Tensor):
                self.dual_coef_ = self.dual_coef_.to(device)
            if self.cv_results_ is not None and isinstance(self.cv_results_, torch.Tensor):
                self.cv_results_ = self.cv_results_.to(device)
            self.device = torch.device(device)
        
        return self
    
    def to_cpu(self, predict_only: bool = False):
        """
        Move parameters to CPU.
        """
        return self.to_device("cpu", predict_only=predict_only)