"""
Takes in an assembly from MTurk, and outputs a .py file that can be directly uploaded to Brain-Score.
Tests are directly generated here as well, and contained in here.
"""

from pathlib import Path


class MTurkBenchmarkFactory:

    def __init__(self, name: str, assembly_name: str, benchmark_directory: str, metric: str, visual_degrees: float,
                 num_trials: int, benchmark_bibtex: str):
        self.name = name
        self.name_lower = name.lower()
        self.assembly_name = assembly_name
        self.benchmark_directory = benchmark_directory
        self.metric = metric
        self.visual_degrees = visual_degrees
        self.num_trials = num_trials
        self.bibtex = benchmark_bibtex

    def __call__(self):

        # create and write benchmark code:
        # benchmark_code = self.generate_benchmark_code()
        # path = Path(f"{self.benchmark_directory}/{self.name_lower}.py")
        # self.write_code_into_file(benchmark_code, path)

        # create and write test code:
        test_code = self.generate_test_code()
        self.write_code_into_file(test_code, Path(f"../../tests/test_benchmarks/test_{self.name_lower}.py"))

    def generate_benchmark_code(self) -> str:
        benchmark_code = f"""
import numpy as np
import brainscore
from brainio.assemblies import walk_coords
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
from brainscore.metrics.error_consistency import ErrorConsistency
from brainscore.model_interface import BrainModel
from brainscore.utils import LazyLoad

BIBTEX = \"\"\"{self.bibtex}\"\"\"
        
        
class _{self.name}{self.metric}(BenchmarkBase):
    # behavioral benchmark
    def __init__(self):
        self._metric = {self.metric}()
        self._assembly = LazyLoad(lambda: load_assembly())
        self._visual_degrees = {self.visual_degrees}
        self._number_of_trials = {self.num_trials}

        super(_{self.name}{self.metric}, self).__init__(
            identifier=f'{self.name}-{self.metric}', version=1,
            ceiling_func=lambda: self._metric.ceiling(self._assembly),
            parent='{self.name}',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        choice_labels = set(self._assembly['truth'].values)
        choice_labels = list(sorted(choice_labels))
        candidate.start_task(BrainModel.Task.label, choice_labels)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        labels = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        raw_score = self._metric(labels, self._assembly)
        ceiling = self.ceiling
        score = raw_score / ceiling.sel(aggregation='center')
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = ceiling
        return score
        
        
def load_assembly():
    assembly = brainscore.get_assembly(f'{self.assembly_name}')
    return assembly
    """

        return benchmark_code

    def write_code_into_file(self, benchmark_code: str, path: Path) -> None:
        with open(path, "w") as f:
            f.write(benchmark_code)

    def generate_test_code(self) -> str:
        test_code = f"""from pathlib import Path
import numpy as np
import pytest
from pytest import approx
from brainio.assemblies import BehavioralAssembly
from brainscore import benchmark_pool
from tests.test_benchmarks import PrecomputedFeatures


class TestBehavioral:
    def test_count(self):
        assert 2 == 2
    
    def test_num_trials(self):
        assert 1 == 1
        """
        return test_code
