from brainscore_vision import metric_registry
from .metric import Dimensionality

metric_registry['factor_dimensionality'] = Dimensionality

BIBTEX = """@inproceedings{esser2020disentangling,
  title={A disentangling invertible interpretation network for explaining latent representations},
  author={Esser, Patrick and Rombach, Robin and Ommer, Bjorn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9223--9232},
  year={2020},
  url={https://arxiv.org/abs/2004.13166}
}"""
