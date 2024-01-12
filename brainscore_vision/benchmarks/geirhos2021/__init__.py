from brainscore_vision import benchmark_registry
from . import benchmark

DATASETS = ['colour', 'contrast', 'cue-conflict', 'edge',
            'eidolonI', 'eidolonII', 'eidolonIII',
            'false-colour', 'high-pass', 'low-pass', 'phase-scrambling', 'power-equalisation',
            'rotation', 'silhouette', 'sketch', 'stylized', 'uniform-noise']

benchmark_registry['Geirhos2021colour-top1'] = getattr(benchmark, "Geirhos2021colourAccuracy")
benchmark_registry['Geirhos2021colour-error_consistency'] = getattr(benchmark, "Geirhos2021colourErrorConsistency")

benchmark_registry['Geirhos2021contrast-top1'] = getattr(benchmark, "Geirhos2021contrastAccuracy")
benchmark_registry['Geirhos2021contrast-error_consistency'] = getattr(benchmark, "Geirhos2021contrastErrorConsistency")

benchmark_registry['Geirhos2021cueconflict-top1'] = getattr(benchmark, "Geirhos2021cueconflictAccuracy")
benchmark_registry['Geirhos2021cueconflict-error_consistency'] = getattr(benchmark, "Geirhos2021cueconflictErrorConsistency")

benchmark_registry['Geirhos2021edge-top1'] = getattr(benchmark, "Geirhos2021edgeAccuracy")
benchmark_registry['Geirhos2021edge-error_consistency'] = getattr(benchmark, "Geirhos2021edgeErrorConsistency")

benchmark_registry['Geirhos2021eidolonI-top1'] = getattr(benchmark, "Geirhos2021eidolonIAccuracy")
benchmark_registry['Geirhos2021eidolonI-error_consistency'] = getattr(benchmark, "Geirhos2021eidolonIErrorConsistency")

benchmark_registry['Geirhos2021eidolonII-top1'] = getattr(benchmark, "Geirhos2021eidolonIIAccuracy")
benchmark_registry['Geirhos2021eidolonII-error_consistency'] = getattr(benchmark, "Geirhos2021eidolonIIErrorConsistency")

benchmark_registry['Geirhos2021eidolonIII-top1'] = getattr(benchmark, "Geirhos2021eidolonIIIAccuracy")
benchmark_registry['Geirhos2021eidolonIII-error_consistency'] = getattr(benchmark, "Geirhos2021eidolonIIIErrorConsistency")

benchmark_registry['Geirhos2021falsecolour-top1'] = getattr(benchmark, "Geirhos2021falsecolourAccuracy")
benchmark_registry['Geirhos2021falsecolour-error_consistency'] = getattr(benchmark, "Geirhos2021falsecolourErrorConsistency")

benchmark_registry['Geirhos2021highpass-top1'] = getattr(benchmark, "Geirhos2021highpassAccuracy")
benchmark_registry['Geirhos2021highpass-error_consistency'] = getattr(benchmark, "Geirhos2021highpassErrorConsistency")

benchmark_registry['Geirhos2021lowpass-top1'] = getattr(benchmark, "Geirhos2021lowpassAccuracy")
benchmark_registry['Geirhos2021lowpass-error_consistency'] = getattr(benchmark, "Geirhos2021lowpassErrorConsistency")

benchmark_registry['Geirhos2021phasescrambling-top1'] = getattr(benchmark, "Geirhos2021phasescramblingAccuracy")
benchmark_registry['Geirhos2021phasescrambling-error_consistency'] = getattr(benchmark, "Geirhos2021phasescramblingErrorConsistency")

benchmark_registry['Geirhos2021powerequalisation-top1'] = getattr(benchmark, "Geirhos2021powerequalisationAccuracy")
benchmark_registry['Geirhos2021powerequalisation-error_consistency'] = getattr(benchmark, "Geirhos2021powerequalisationErrorConsistency")

benchmark_registry['Geirhos2021rotation-top1'] = getattr(benchmark, "Geirhos2021rotationAccuracy")
benchmark_registry['Geirhos2021rotation-error_consistency'] = getattr(benchmark, "Geirhos2021rotationErrorConsistency")

benchmark_registry['Geirhos2021silhouette-top1'] = getattr(benchmark, "Geirhos2021silhouetteAccuracy")
benchmark_registry['Geirhos2021silhouette-error_consistency'] = getattr(benchmark, "Geirhos2021silhouetteErrorConsistency")

benchmark_registry['Geirhos2021sketch-top1'] = getattr(benchmark, "Geirhos2021sketchAccuracy")
benchmark_registry['Geirhos2021sketch-error_consistency'] = getattr(benchmark, "Geirhos2021sketchErrorConsistency")

benchmark_registry['Geirhos2021stylized-top1'] = getattr(benchmark, "Geirhos2021stylizedAccuracy")
benchmark_registry['Geirhos2021stylized-error_consistency'] = getattr(benchmark, "Geirhos2021stylizedErrorConsistency")

benchmark_registry['Geirhos2021uniformnoise-top1'] = getattr(benchmark, "Geirhos2021uniformnoiseAccuracy")
benchmark_registry['Geirhos2021uniformnoise-error_consistency'] = getattr(benchmark, "Geirhos2021uniformnoiseErrorConsistency")

