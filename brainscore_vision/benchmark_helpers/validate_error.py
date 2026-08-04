"""Submission/CI validation that a benchmark reports usable uncertainty.

The gate is deliberately lenient during transition: a score may report ``error = nan``
*if* it declares ``error_nan_reason``. New benchmarks should additionally preserve the
disaggregated ``raw`` values so true split-half reliability stays computable later.
"""

import math
from dataclasses import dataclass, field
from typing import List

from brainscore_core import Score

from brainscore_vision.metric_helpers.bootstrap_error import (
    ERROR_KEY,
    ERROR_NAN_REASON_KEY,
    ERROR_OVER_KEY,
)


@dataclass
class ValidationResult:
    ok: bool
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_score_uncertainty(
    score: "Score",
    require_raw: bool = True,
    allow_nan_with_reason: bool = True,
) -> ValidationResult:
    """Check that ``score`` carries a usable error, on the score scale, with provenance.

    :param require_raw: require the disaggregated ``raw`` values (recommended for new
        benchmarks; lets the platform compute true split-half reliability later).
    :param allow_nan_with_reason: permit ``error = nan`` when ``error_nan_reason`` is set.
    """
    problems: List[str] = []
    warnings: List[str] = []
    attrs = score.attrs

    if ERROR_KEY not in attrs:
        problems.append("score.attrs is missing 'error'")
        return ValidationResult(ok=False, problems=problems, warnings=warnings)

    error = attrs[ERROR_KEY]
    try:
        error = float(error)
    except (TypeError, ValueError):
        problems.append(f"'error' is not a float: {error!r}")
        return ValidationResult(ok=False, problems=problems, warnings=warnings)

    if math.isnan(error):
        if not (allow_nan_with_reason and attrs.get(ERROR_NAN_REASON_KEY)):
            problems.append(
                "'error' is nan without an 'error_nan_reason'; resample stimuli/subjects "
                "to estimate it, or declare why it cannot be estimated"
            )
    else:
        if not math.isfinite(error):
            problems.append(f"'error' is not finite: {error}")
        if error < 0:
            problems.append(f"'error' is negative: {error}")
        center = float(score.values) if score.ndim == 0 else float("nan")
        if math.isfinite(center) and error > max(1.0, 2 * abs(center)):
            warnings.append(
                f"'error' ({error:.3f}) is large vs the score ({center:.3f}); "
                "check it is on the same (ceiled) scale as the score"
            )
        if not attrs.get(ERROR_OVER_KEY):
            warnings.append(
                "missing 'error_over' metadata; record which units were resampled "
                "(e.g. ['stimulus', 'subject']) so errors are comparable across benchmarks"
            )

    if require_raw and Score.RAW_VALUES_KEY not in attrs:
        problems.append(
            "missing disaggregated 'raw' values; preserve per-stimulus/per-subject scores "
            "so true split-half reliability is computable without re-running the model"
        )

    return ValidationResult(ok=not problems, problems=problems, warnings=warnings)


def assert_valid_uncertainty(score: "Score", **kwargs) -> None:
    """Pytest/CI entry point: raise on any problem, surfacing all of them at once."""
    result = validate_score_uncertainty(score, **kwargs)
    for warning in result.warnings:
        print(f"[uncertainty][warn] {warning}")
    assert result.ok, "uncertainty validation failed:\n  - " + "\n  - ".join(result.problems)
