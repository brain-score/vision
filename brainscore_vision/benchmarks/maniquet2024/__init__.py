from brainscore_vision import benchmark_registry
from .benchmark import Maniquet2024ConfusionSimilarity, Maniquet2024TasksConsistency

benchmark_registry['Maniquet2024ConfusionSimilarity'] = lambda: Maniquet2024ConfusionSimilarity()
benchmark_registry['Maniquet2024TasksConsistency'] = lambda: Maniquet2024TasksConsistency()

