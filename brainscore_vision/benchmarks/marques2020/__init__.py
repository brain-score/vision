from brainscore_vision import benchmark_registry
from .benchmarks.cavanaugh2002a_benchmark import MarquesCavanaugh2002V1SurroundSuppressionIndex, \
    MarquesCavanaugh2002V1GratingSummationField, MarquesCavanaugh2002V1SurroundDiameter

benchmark_registry['Marques2020_Cavanaugh2002-grating_summation_field'] = \
    MarquesCavanaugh2002V1GratingSummationField
benchmark_registry['Marques2020_Cavanaugh2002-surround_diameter'] = \
    MarquesCavanaugh2002V1SurroundDiameter
benchmark_registry['Marques2020_Cavanaugh2002-surround_suppression_index'] = \
    MarquesCavanaugh2002V1SurroundSuppressionIndex


from .benchmarks.devalois1982a_benchmark import MarquesDeValois1982V1PreferredOrientation

benchmark_registry['Marques2020_DeValois1982-pref_or'] = \
    MarquesDeValois1982V1PreferredOrientation


from .benchmarks.devalois1982b_benchmark import MarquesDeValois1982V1PeakSpatialFrequency

benchmark_registry['Marques2020_DeValois1982-peak_sf'] = \
    MarquesDeValois1982V1PeakSpatialFrequency


from .benchmarks.freemanZiemba2013_benchmark import MarquesFreemanZiemba2013V1TextureModulationIndex, \
    MarquesFreemanZiemba2013V1AbsoluteTextureModulationIndex, MarquesFreemanZiemba2013V1TextureSelectivity, \
    MarquesFreemanZiemba2013V1TextureSparseness, MarquesFreemanZiemba2013V1VarianceRatio, \
    MarquesFreemanZiemba2013V1MaxTexture, MarquesFreemanZiemba2013V1MaxNoise

# V1 properties benchmarks: texture modulation
benchmark_registry['Marques2020_FreemanZiemba2013-texture_modulation_index'] = \
    MarquesFreemanZiemba2013V1TextureModulationIndex
benchmark_registry['Marques2020_FreemanZiemba2013-abs_texture_modulation_index'] = \
    MarquesFreemanZiemba2013V1AbsoluteTextureModulationIndex

# V1 properties benchmarks: selectivity
benchmark_registry['Marques2020_FreemanZiemba2013-texture_selectivity'] = \
    MarquesFreemanZiemba2013V1TextureSelectivity
benchmark_registry['Marques2020_FreemanZiemba2013-texture_sparseness'] = \
    MarquesFreemanZiemba2013V1TextureSparseness
benchmark_registry['Marques2020_FreemanZiemba2013-texture_variance_ratio'] = \
    MarquesFreemanZiemba2013V1VarianceRatio

# V1 properties benchmarks: magnitude
benchmark_registry['Marques2020_FreemanZiemba2013-max_texture'] = MarquesFreemanZiemba2013V1MaxTexture
benchmark_registry['Marques2020_FreemanZiemba2013-max_noise'] = MarquesFreemanZiemba2013V1MaxNoise


from .benchmarks.ringach2002_benchmark import MarquesRingach2002V1CircularVariance, MarquesRingach2002V1Bandwidth, \
        MarquesRingach2002V1OrthogonalPreferredRatio, MarquesRingach2002V1OrientationSelective, \
        MarquesRingach2002V1CircularVarianceBandwidthRatio, \
        MarquesRingach2002V1OrthogonalPrefferredRatioCircularVarianceDifference, MarquesRingach2002V1MaxDC, \
        MarquesRingach2002V1ModulationRatio

# V1 properties benchmarks: orientation
benchmark_registry['Marques2020_Ringach2002-circular_variance'] = MarquesRingach2002V1CircularVariance
benchmark_registry['Marques2020_Ringach2002-or_bandwidth'] = MarquesRingach2002V1Bandwidth
benchmark_registry['Marques2020_Ringach2002-orth_pref_ratio'] = MarquesRingach2002V1OrthogonalPreferredRatio
benchmark_registry['Marques2020_Ringach2002-or_selective'] = MarquesRingach2002V1OrientationSelective
benchmark_registry['Marques2020_Ringach2002-cv_bandwidth_ratio'] = \
        MarquesRingach2002V1CircularVarianceBandwidthRatio
benchmark_registry['Marques2020_Ringach2002-opr_cv_diff'] = \
        MarquesRingach2002V1OrthogonalPrefferredRatioCircularVarianceDifference

# V1 properties benchmarks: magnitude
benchmark_registry['Marques2020_Ringach2002-max_dc'] = MarquesRingach2002V1MaxDC
benchmark_registry['Marques2020_Ringach2002-modulation_ratio'] = MarquesRingach2002V1ModulationRatio


from .benchmarks.schiller1976_benchmark import MarquesSchiller1976V1SpatialFrequencyBandwidth, \
        MarquesSchiller1976V1SpatialFrequencySelective

# V1 properties benchmarks: spatial frequency
benchmark_registry['Marques2020_Schiller1976-sf_selective'] = MarquesSchiller1976V1SpatialFrequencySelective
benchmark_registry['Marques2020_Schiller1976-sf_bandwidth'] = MarquesSchiller1976V1SpatialFrequencyBandwidth

