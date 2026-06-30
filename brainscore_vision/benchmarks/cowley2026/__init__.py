from brainscore_vision import benchmark_registry


BIBTEX = """@article{cowley2026compact,
  title={Compact deep neural network models of the visual cortex},
  author={Cowley, Benjamin R and Stan, Patricia L and Pillow, Jonathan W and Smith, Matthew A},
  journal={Nature},
  volume={652},
  number={8111},
  pages={947--954},
  year={2026},
  publisher={Nature Publishing Group}}"""


from .benchmarks.benchmark_Cowley2026_190923 import benchmark_Cowley2026_190923
benchmark_registry['Cowley2026_190923'] = benchmark_Cowley2026_190923



