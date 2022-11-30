from brainscore_vision import benchmark_registry
from .benchmark import MarquesCavanaugh2002V1SurroundSuppressionIndex, MarquesCavanaugh2002V1GratingSummationField, \
    MarquesCavanaugh2002V1SurroundDiameter

benchmark_registry['dicarlo.Marques2020_Cavanaugh2002-grating_summation_field'] = \
    MarquesCavanaugh2002V1GratingSummationField
benchmark_registry['dicarlo.Marques2020_Cavanaugh2002-surround_diameter'] = \
    MarquesCavanaugh2002V1SurroundDiameter
benchmark_registry['dicarlo.Marques2020_Cavanaugh2002-surround_suppression_index'] = \
    MarquesCavanaugh2002V1SurroundSuppressionIndex
