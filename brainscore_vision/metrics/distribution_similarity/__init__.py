from brainscore_vision import metric_registry
from .metric import BootstrapDistributionSimilarity, NeuronalPropertyCeiling

metric_registry['ks_similarity'] = BootstrapDistributionSimilarity
metric_registry['property_ceiling'] = NeuronalPropertyCeiling

BIBTEX = """@article{marques2021multi,
  title={Multi-scale hierarchical neural network models that bridge from single neurons in the primate primary visual cortex to object recognition behavior},
  author={Marques, Tiago and Schrimpf, Martin and DiCarlo, James J},
  journal={bioRxiv},
  pages={2021--03},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}"""
