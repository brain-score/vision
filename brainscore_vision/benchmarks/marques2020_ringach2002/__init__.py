from brainscore_vision import benchmark_registry
from .benchmark import MarquesRingach2002V1CircularVariance, MarquesRingach2002V1Bandwidth, \
        MarquesRingach2002V1OrthogonalPreferredRatio, MarquesRingach2002V1OrientationSelective, \
        MarquesRingach2002V1CircularVarianceBandwidthRatio, \
        MarquesRingach2002V1OrthogonalPrefferredRatioCircularVarianceDifference, MarquesRingach2002V1MaxDC, \
        MarquesRingach2002V1ModulationRatio

# V1 properties benchmarks: orientation
benchmark_registry['dicarlo.Marques2020_Ringach2002-circular_variance'] = MarquesRingach2002V1CircularVariance
benchmark_registry['dicarlo.Marques2020_Ringach2002-or_bandwidth'] = MarquesRingach2002V1Bandwidth
benchmark_registry['dicarlo.Marques2020_Ringach2002-orth_pref_ratio'] = MarquesRingach2002V1OrthogonalPreferredRatio
benchmark_registry['dicarlo.Marques2020_Ringach2002-or_selective'] = MarquesRingach2002V1OrientationSelective
benchmark_registry['dicarlo.Marques2020_Ringach2002-cv_bandwidth_ratio'] = \
        MarquesRingach2002V1CircularVarianceBandwidthRatio
benchmark_registry['dicarlo.Marques2020_Ringach2002-opr_cv_diff'] = \
        MarquesRingach2002V1OrthogonalPrefferredRatioCircularVarianceDifference

# V1 properties benchmarks: magnitude
benchmark_registry['dicarlo.Marques2020_Ringach2002-max_dc'] = MarquesRingach2002V1MaxDC
benchmark_registry['dicarlo.Marques2020_Ringach2002-modulation_ratio'] = MarquesRingach2002V1ModulationRatio