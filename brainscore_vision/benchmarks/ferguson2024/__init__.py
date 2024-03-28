from brainscore_vision import benchmark_registry
from benchmarks.ferguson2024.benchmark import Ferguson2024AlignmentMeasure

DATASETS = ['circle_line', 'color', 'convergence', 'eighth',
            'gray_easy', 'gray_hard', 'half', 'juncture',
            'lle', 'llh', 'quarter', 'round_f',
            'round_v', 'tilted_line']

benchmark_registry['Ferguson2024circle_line-alignment_measure'] = Ferguson2024AlignmentMeasure("circle_line")
benchmark_registry['Ferguson2024color-alignment_measure'] = Ferguson2024AlignmentMeasure("color")
benchmark_registry['Ferguson2024convergence-alignment_measure'] = Ferguson2024AlignmentMeasure("convergence")
benchmark_registry['Ferguson2024eighth-alignment_measure'] = Ferguson2024AlignmentMeasure("eighth")
benchmark_registry['Ferguson2024gray_easy-alignment_measure'] = Ferguson2024AlignmentMeasure("gray_easy")
benchmark_registry['Ferguson2024gray_hard-alignment_measure'] = Ferguson2024AlignmentMeasure("gray_hard")
benchmark_registry['Ferguson2024half-alignment_measure'] = Ferguson2024AlignmentMeasure("half")
benchmark_registry['Ferguson2024juncture-alignment_measure'] = Ferguson2024AlignmentMeasure("juncture")
benchmark_registry['Ferguson2024lle-alignment_measure'] = Ferguson2024AlignmentMeasure("lle")
benchmark_registry['Ferguson2024llh-alignment_measure'] = Ferguson2024AlignmentMeasure("llh")
benchmark_registry['Ferguson2024quarter-alignment_measure'] = Ferguson2024AlignmentMeasure("quarter")
benchmark_registry['Ferguson2024round_f-alignment_measure'] = Ferguson2024AlignmentMeasure("round_f")
benchmark_registry['Ferguson2024round_v-alignment_measure'] = Ferguson2024AlignmentMeasure("round_v")
benchmark_registry['Ferguson2024tilted_line-alignment_measure'] = Ferguson2024AlignmentMeasure("tilted_line")


