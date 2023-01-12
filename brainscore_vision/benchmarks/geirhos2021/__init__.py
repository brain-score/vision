from brainscore_vision import benchmark_registry
from . import benchmark

DATASETS = ['colour', 'contrast', 'cue-conflict', 'edge',
            'eidolonI', 'eidolonII', 'eidolonIII',
            'false-colour', 'high-pass', 'low-pass', 'phase-scrambling', 'power-equalisation',
            'rotation', 'silhouette', 'sketch', 'stylized', 'uniform-noise']

benchmark_ctr = getattr(benchmark, "Geirhos2021colourAccuracy")
benchmark_registry['brendel.Geirhos2021colour-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021colourErrorConsistency")
benchmark_registry['brendel.Geirhos2021colour-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021contrastAccuracy")
benchmark_registry['brendel.Geirhos2021contrast-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021contrastErrorConsistency")
benchmark_registry['brendel.Geirhos2021contrast-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021cueconflictAccuracy")
benchmark_registry['brendel.Geirhos2021cueconflict-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021cueconflictErrorConsistency")
benchmark_registry['brendel.Geirhos2021cueconflict-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021edgeAccuracy")
benchmark_registry['brendel.Geirhos2021edge-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021edgeErrorConsistency")
benchmark_registry['brendel.Geirhos2021edge-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021eidolonIAccuracy")
benchmark_registry['brendel.Geirhos2021eidolonI-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021eidolonIErrorConsistency")
benchmark_registry['brendel.Geirhos2021eidolonI-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021eidolonIIAccuracy")
benchmark_registry['brendel.Geirhos2021eidolonII-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021eidolonIIErrorConsistency")
benchmark_registry['brendel.Geirhos2021eidolonII-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021eidolonIIIAccuracy")
benchmark_registry['brendel.Geirhos2021eidolonIII-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021eidolonIIIErrorConsistency")
benchmark_registry['brendel.Geirhos2021eidolonIII-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021falsecolourAccuracy")
benchmark_registry['brendel.Geirhos2021falsecolour-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021falsecolourErrorConsistency")
benchmark_registry['brendel.Geirhos2021falsecolour-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021highpassAccuracy")
benchmark_registry['brendel.Geirhos2021highpass-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021highpassErrorConsistency")
benchmark_registry['brendel.Geirhos2021highpass-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021lowpassAccuracy")
benchmark_registry['brendel.Geirhos2021lowpass-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021lowpassErrorConsistency")
benchmark_registry['brendel.Geirhos2021lowpass-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021phasescramblingAccuracy")
benchmark_registry['brendel.Geirhos2021phasescrambling-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021phasescramblingErrorConsistency")
benchmark_registry['brendel.Geirhos2021phasescrambling-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021powerequalisationAccuracy")
benchmark_registry['brendel.Geirhos2021powerequalisation-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021powerequalisationErrorConsistency")
benchmark_registry['brendel.Geirhos2021powerequalisation-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021rotationAccuracy")
benchmark_registry['brendel.Geirhos2021rotation-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021rotationErrorConsistency")
benchmark_registry['brendel.Geirhos2021rotation-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021silhouetteAccuracy")
benchmark_registry['brendel.Geirhos2021silhouette-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021silhouetteErrorConsistency")
benchmark_registry['brendel.Geirhos2021silhouette-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021sketchAccuracy")
benchmark_registry['brendel.Geirhos2021sketch-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021sketchErrorConsistency")
benchmark_registry['brendel.Geirhos2021sketch-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021stylizedAccuracy")
benchmark_registry['brendel.Geirhos2021stylized-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021stylizedErrorConsistency")
benchmark_registry['brendel.Geirhos2021stylized-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

benchmark_ctr = getattr(benchmark, "Geirhos2021uniformnoiseAccuracy")
benchmark_registry['brendel.Geirhos2021uniformnoise-top1'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()
benchmark_ctr = getattr(benchmark, "Geirhos2021uniformnoiseErrorConsistency")
benchmark_registry['brendel.Geirhos2021uniformnoise-error_consistency'] = lambda benchmark_ctr=benchmark_ctr: benchmark_ctr()

