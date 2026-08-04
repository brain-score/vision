"""Shared utilities for attaching a resampling-based standard error to a Score.

Proposed home: ``brainscore_vision/metric_helpers/bootstrap_error.py``.

A benchmark's reported score is one realisation drawn over a finite set of stimuli
and subjects. The ``error`` attribute records how much that score would move under
resampling of those units, on the same scale as the score. It is the ingredient the
platform needs to estimate per-benchmark model-ranking reliability
(``1 - mean(error**2) / var(scores across models)``) without re-running any model.

This complements :func:`~brainscore_vision.metric_helpers.transformations.standard_error_of_the_mean`
(the analytic SE across an existing ``split`` dimension); use that when the metric
already aggregates over cross-validation splits, and :func:`bootstrap_error` for the
general case (single aggregate, behavioural metrics, multi-unit designs).
"""

import logging
from typing import Callable, Optional, Sequence

import numpy as np

from brainscore_core import Score

logger = logging.getLogger(__name__)

ERROR_KEY = "error"
ERROR_OVER_KEY = "error_over"
N_BOOTSTRAP_KEY = "n_bootstrap"
ERROR_NAN_REASON_KEY = "error_nan_reason"

Aggregate = Callable[["Score"], float]


def _full_mean(values: "Score") -> float:
    return float(np.nanmean(np.asarray(values)))


def _resample(values: "Score", over: Sequence[str], rng: np.random.Generator) -> "Score":
    """Resample ``values`` with replacement along each dimension in ``over``.

    Dimensions are resampled independently (crossed design: every subject sees every
    stimulus), which is the Brain-Score default. For genuinely nested designs, resample
    the outer unit and pass only that dimension, then bootstrap the inner unit within.
    """
    index = {dim: rng.integers(0, values.sizes[dim], values.sizes[dim]) for dim in over}
    return values.isel(index)


def bootstrap_error(
    values: "Score",
    over: Sequence[str],
    aggregate: Optional[Aggregate] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> float:
    """Standard error of the aggregated score from resampling ``over`` units.

    :param values: the disaggregated per-unit scores (e.g. per-stimulus correlations,
        per-subject-and-condition consistencies) with ``over`` among its dimensions.
    :param over: the sampling dimensions to resample with replacement, e.g.
        ``["stimulus"]`` or ``["subject", "stimulus"]``.
    :param aggregate: maps the resampled assembly to the scalar score; defaults to the
        nan-mean over all dimensions. Pass the metric's own aggregation when it is not a
        plain mean (e.g. median over neuroids then mean over stimuli).
    :param n_bootstrap: number of resamples.
    :param seed: fixed for reproducibility; never use unseeded randomness.
    :return: the standard deviation of the bootstrap score distribution. For a plain
        mean over ``n`` units this matches ``std / sqrt(n)``.
    """
    missing = [dim for dim in over if dim not in values.dims]
    if missing:
        raise ValueError(f"resample dims {missing} not in values.dims {list(values.dims)}")
    aggregate = aggregate or _full_mean
    rng = np.random.default_rng(seed)
    draws = np.array(
        [aggregate(_resample(values, over, rng)) for _ in range(n_bootstrap)]
    )
    draws = draws[np.isfinite(draws)]
    if len(draws) < 2:
        logger.warning("bootstrap produced <2 finite draws; returning nan error")
        return float("nan")
    return float(np.std(draws, ddof=1))


def attach_error(
    score: "Score",
    values: "Score",
    over: Sequence[str],
    aggregate: Optional[Aggregate] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> "Score":
    """Compute the bootstrap error and attach it, with metadata, to ``score``.

    Sets ``score.attrs['error']`` plus ``error_over`` / ``n_bootstrap`` provenance, and
    preserves the disaggregated ``values`` under ``score.attrs['raw']`` so the platform
    can later compute true split-half reliability without re-running the model.
    """
    score.attrs[ERROR_KEY] = bootstrap_error(values, over, aggregate, n_bootstrap, seed)
    score.attrs[ERROR_OVER_KEY] = list(over)
    score.attrs[N_BOOTSTRAP_KEY] = int(n_bootstrap)
    if Score.RAW_VALUES_KEY not in score.attrs:
        score.attrs[Score.RAW_VALUES_KEY] = values
    return score


def declare_no_error(score: "Score", reason: str) -> "Score":
    """Explicitly mark a score as non-resampleable instead of silently omitting error.

    Use only when no unit can be resampled (e.g. a single deterministic comparison).
    Records ``error = nan`` and a human-readable ``error_nan_reason`` so validation can
    distinguish 'cannot estimate' from 'forgot to estimate'.
    """
    score.attrs[ERROR_KEY] = float("nan")
    score.attrs[ERROR_NAN_REASON_KEY] = reason
    return score
