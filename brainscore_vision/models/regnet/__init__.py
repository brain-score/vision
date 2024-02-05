from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, LAYERS

BIBTEX = """@inproceedings{radosavovic2020designing,
  title={Designing network design spaces},
  author={Radosavovic, Ilija and Kosaraju, Raj Prateek and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={10428--10436},
  year={2020}
}"""

model_registry['regnet_y_400mf'] = lambda: ModelCommitment(
    identifier='regnet_y_400mf', activations_model=get_model(), layers=LAYERS)
