from brainscore_vision import benchmark_registry
from .benchmark import PhysionGlobalPredictionHumanCohenK
from .benchmark import PhysionGlobalDetectionHumanCohenK
from .benchmark import PhysionSnippetDetectionHumanCohenK
from .benchmark import PhysionSnippetDetectionIntraScenarioHumanCohenK
from .benchmark import PhysionGlobalPredictionIntraScenarioHumanCohenK
from .benchmark import PhysionGlobalDetectionIntraScenarioHumanCohenK

benchmark_registry['Physionv1.5-ocd-cohenk'] = PhysionGlobalDetectionHumanCohenK
benchmark_registry['Physionv1.5-ocp-cohenk'] = PhysionGlobalPredictionHumanCohenK
benchmark_registry['Physionv1.5-snippet-rollout-cohenk'] = PhysionSnippetDetectionHumanCohenK
benchmark_registry['Physionv1.5-snippet-rollout-intra-cohenk'] =PhysionSnippetDetectionIntraScenarioHumanCohenK
benchmark_registry['Physionv1.5-ocp-intra-generalization-cohenk'] = PhysionGlobalPredictionIntraScenarioHumanCohenK
benchmark_registry['Physionv1.5-ocd-intra-generalization-cohenk'] = PhysionGlobalDetectionIntraScenarioHumanCohenK