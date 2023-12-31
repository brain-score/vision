from brainscore_vision import metric_registry
from brainscore_vision.metrics.regression_correlation.metric import CrossRegressedCorrelation, pearsonr_correlation
from .metric import mask_regression

metric_registry['mask_regression'] = lambda *args, **kwargs: CrossRegressedCorrelation(
    regression=mask_regression(),
    correlation=pearsonr_correlation(),
    *args, **kwargs)

BIBTEX = """@article{klindt2017neural,
  title={Neural system identification for large populations separating “what” and “where”},
  author={Klindt, David and Ecker, Alexander S and Euler, Thomas and Bethge, Matthias},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}"""
