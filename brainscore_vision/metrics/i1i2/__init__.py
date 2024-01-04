from brainscore_vision import metric_registry
from .metric import I1, I2n

metric_registry['i1'] = I1
metric_registry['i2n'] = I2n

BIBTEX = """@article{rajalingham2018large,
  title={Large-scale, high-resolution comparison of the core visual object recognition behavior of humans, monkeys, and state-of-the-art deep artificial neural networks},
  author={Rajalingham, Rishi and Issa, Elias B and Bashivan, Pouya and Kar, Kohitij and Schmidt, Kailyn and DiCarlo, James J},
  journal={Journal of Neuroscience},
  volume={38},
  number={33},
  pages={7255--7269},
  year={2018},
  publisher={Soc Neuroscience}
}"""
