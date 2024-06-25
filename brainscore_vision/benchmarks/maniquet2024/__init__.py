from brainscore_vision import benchmark_registry
from .benchmark import Maniquet2024ConfusionSimilarity

benchmark_registry['Maniquet2024ConfusionSimilarity'] = lambda: Maniquet2024ConfusionSimilarity()

