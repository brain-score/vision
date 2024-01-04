from brainscore_vision import metric_registry
from .metric import CKA, CKACrossValidated

metric_registry['cka'] = CKA
metric_registry['cka_cv'] = CKACrossValidated

BIBTEX = """@inproceedings{kornblith2019similarity,
  title={Similarity of neural network representations revisited},
  author={Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey},
  booktitle={International conference on machine learning},
  pages={3519--3529},
  year={2019},
  organization={PMLR}
}"""
