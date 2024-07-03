from brainscore_vision import benchmark_registry
from .benchmark import Maniquet2024ConfusionSimilarity, Maniquet2024TasksConsistency

benchmark_registry['Maniquet2024-confusion_similarity'] = lambda: Maniquet2024ConfusionSimilarity()
benchmark_registry['Maniquet2024-tasks_consistency'] = lambda: Maniquet2024TasksConsistency()

