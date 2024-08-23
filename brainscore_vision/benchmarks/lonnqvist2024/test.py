import pytest
from pytest import approx

from brainscore_vision import benchmark_registry, load_benchmark


# TODO: ceilings


@pytest.mark.parametrize('benchmark', [
    'Lonnqvist2024_InlabInstructionsBehavioralAccuracyDistance',
    'Lonnqvist2024_InlabNoInstructionsBehavioralAccuracyDistance',
    'Lonnqvist2024_InlabInstructionsBehavioralAccuracyDistance',
    'Lonnqvist2024_EngineeringAccuracy',
])
def test_benchmark_registry(benchmark):
    assert benchmark in benchmark_registry


class TestBehavioral:
    @pytest.mark.private_access
    @pytest.mark.parametrize('dataset, expected_ceiling', [
        ('InlabInstructionsBehavioralAccuracyDistance', approx(0.0, abs=0.001)),
        ('InlabNoInstructionsBehavioralAccuracyDistance', approx(0.0, abs=0.001)),
        ('InlabInstructionsBehavioralAccuracyDistance', approx(0.0, abs=0.001)),
    ])
    def test_dataset_ceiling(self, dataset, expected_ceiling):
        benchmark = f"Lonnqvist2024_{dataset}BehavioralAccuracyDistance"
        benchmark = load_benchmark(benchmark)
        ceiling = benchmark.ceiling
        assert ceiling == expected_ceiling

    @pytest.mark.private_access
    @pytest.mark.parametrize('dataset, expected_raw_score', [
        ('InlabInstructionsBehavioralAccuracyDistance', approx(0.0, abs=0.001)),
        ('InlabNoInstructionsBehavioralAccuracyDistance', approx(0.0, abs=0.001)),
        ('InlabInstructionsBehavioralAccuracyDistance', approx(0.0, abs=0.001)),
    ])
    def test_model(self, dataset, expected_raw_score):
        if 'all' in dataset:
            benchmark = f"Scialom2024_{dataset}BehavioralErrorConsistency"
        else:
            benchmark = f"Scialom2024_{dataset}BehavioralAccuracyDistance"
        benchmark = load_benchmark(benchmark)
        model = 'alexnet'
        score = benchmark(model)
        raw_score = score.raw
        # division by ceiling <= 1 should result in higher score
        assert score >= raw_score
        assert raw_score == expected_raw_score


class TestEngineering:
    @pytest.mark.parametrize('dataset, expected_accuracy', [
        ('EngineeringAccuracy', approx(0.0, abs=0.001)),
    ])
    def test_accuracy(self, dataset, expected_accuracy):
        benchmark = load_benchmark(f"Lonnqvist2024_{dataset}")
        model = 'alexnet'
        score = benchmark(model)
        raw_score = score.raw
        # division by ceiling <= 1 should result in higher score
        assert score >= raw_score
        assert raw_score == expected_accuracy
