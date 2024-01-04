from brainscore_vision import metric_registry
from .metric import RDMMetric, RDMCrossValidated

metric_registry['rdm'] = RDMMetric
metric_registry['rdm_cv'] = RDMCrossValidated

BIBTEX = """@article{kriegeskorte2008representational,
  title={Representational similarity analysis-connecting the branches of systems neuroscience},
  author={Kriegeskorte, Nikolaus and Mur, Marieke and Bandettini, Peter A},
  journal={Frontiers in systems neuroscience},
  pages={4},
  year={2008},
  publisher={Frontiers}
}"""
