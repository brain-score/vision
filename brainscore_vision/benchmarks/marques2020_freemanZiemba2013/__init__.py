from brainscore_vision import benchmark_registry
from .benchmark import MarquesFreemanZiemba2013V1TextureModulationIndex, \
    MarquesFreemanZiemba2013V1AbsoluteTextureModulationIndex, MarquesFreemanZiemba2013V1TextureSelectivity, \
    MarquesFreemanZiemba2013V1TextureSparseness, MarquesFreemanZiemba2013V1VarianceRatio, \
    MarquesFreemanZiemba2013V1MaxTexture, MarquesFreemanZiemba2013V1MaxNoise

# V1 properties benchmarks: texture modulation
benchmark_registry['dicarlo.Marques2020_FreemanZiemba2013-texture_modulation_index'] = \
    MarquesFreemanZiemba2013V1TextureModulationIndex
benchmark_registry['dicarlo.Marques2020_FreemanZiemba2013-abs_texture_modulation_index'] = \
    MarquesFreemanZiemba2013V1AbsoluteTextureModulationIndex

# V1 properties benchmarks: selectivity
benchmark_registry['dicarlo.Marques2020_FreemanZiemba2013-texture_selectivity'] = \
    MarquesFreemanZiemba2013V1TextureSelectivity
benchmark_registry['dicarlo.Marques2020_FreemanZiemba2013-texture_sparseness'] = \
    MarquesFreemanZiemba2013V1TextureSparseness
benchmark_registry['dicarlo.Marques2020_FreemanZiemba2013-texture_variance_ratio'] = \
    MarquesFreemanZiemba2013V1VarianceRatio

# V1 properties benchmarks: magnitude
benchmark_registry['dicarlo.Marques2020_FreemanZiemba2013-max_texture'] = MarquesFreemanZiemba2013V1MaxTexture
benchmark_registry['dicarlo.Marques2020_FreemanZiemba2013-max_noise'] = MarquesFreemanZiemba2013V1MaxNoise