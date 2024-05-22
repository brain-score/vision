from brainscore_vision import benchmark_registry
from benchmarks.ferguson2024.benchmark import Ferguson2024ValueDelta

DATASETS = ['circle_line', 'color', 'convergence', 'eighth',
            'gray_easy', 'gray_hard', 'half', 'juncture',
            'lle', 'llh', 'quarter', 'round_f',
            'round_v', 'tilted_line']

benchmark_registry['Ferguson2024circle_line-value_delta'] = Ferguson2024ValueDelta("circle_line")
benchmark_registry['Ferguson2024color-value_delta'] = Ferguson2024ValueDelta("color")
benchmark_registry['Ferguson2024convergence-value_delta'] = Ferguson2024ValueDelta("convergence")
benchmark_registry['Ferguson2024eighth-value_delta'] = Ferguson2024ValueDelta("eighth")
benchmark_registry['Ferguson2024gray_easy-value_delta'] = Ferguson2024ValueDelta("gray_easy")
benchmark_registry['Ferguson2024gray_hard-value_delta'] = Ferguson2024ValueDelta("gray_hard")
benchmark_registry['Ferguson2024half-value_delta'] = Ferguson2024ValueDelta("half")
benchmark_registry['Ferguson2024juncture-value_delta'] = Ferguson2024ValueDelta("juncture")
benchmark_registry['Ferguson2024lle-value_delta'] = Ferguson2024ValueDelta("lle")
benchmark_registry['Ferguson2024llh-value_delta'] = Ferguson2024ValueDelta("llh")
benchmark_registry['Ferguson2024quarter-value_delta'] = Ferguson2024ValueDelta("quarter")
benchmark_registry['Ferguson2024round_f-value_delta'] = Ferguson2024ValueDelta("round_f")
benchmark_registry['Ferguson2024round_v-value_delta'] = Ferguson2024ValueDelta("round_v")
benchmark_registry['Ferguson2024tilted_line-value_delta'] = Ferguson2024ValueDelta("tilted_line")


