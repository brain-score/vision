from brainscore_vision import benchmark_registry
from .benchmark import _Islam2021Dimensionality

benchmark_registry['Islam2021-shape_v1_dimensionality'] = lambda: _Islam2021Dimensionality("V1", "shape")
benchmark_registry['Islam2021-texture_v1_dimensionality'] = lambda: _Islam2021Dimensionality("V1", "texture")
benchmark_registry['Islam2021-shape_v2_dimensionality'] = lambda: _Islam2021Dimensionality("V2", "shape")
benchmark_registry['Islam2021-texture_v2_dimensionality'] = lambda: _Islam2021Dimensionality("V2", "texture")
benchmark_registry['Islam2021-shape_v4_dimensionality'] = lambda: _Islam2021Dimensionality("V4", "shape")
benchmark_registry['Islam2021-texture_v4_dimensionality'] = lambda: _Islam2021Dimensionality("V4", "texture")
benchmark_registry['Islam2021-shape_it_dimensionality'] = lambda: _Islam2021Dimensionality("IT", "shape")
benchmark_registry['Islam2021-texture_it_dimensionality'] = lambda: _Islam2021Dimensionality("IT", "texture")
