from brainscore_vision import metric_registry
from .metric import ErrorConsistency

metric_registry['error_consistency'] = ErrorConsistency

BIBTEX = """@article{geirhos2020beyond,
  title={Beyond accuracy: quantifying trial-by-trial behaviour of CNNs and humans by measuring error consistency},
  author={Geirhos, Robert and Meding, Kristof and Wichmann, Felix A},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={13890--13902},
  year={2020}
}"""
