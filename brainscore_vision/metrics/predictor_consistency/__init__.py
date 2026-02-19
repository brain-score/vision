from brainscore_vision import metric_registry
from .ceiling import SplitHalfPredictorConsistency

metric_registry["predictor_consistency"] = lambda *a, **k: SplitHalfPredictorConsistency()
