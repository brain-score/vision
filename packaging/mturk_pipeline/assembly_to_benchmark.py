"""
Takes in an assembly from MTurk, and outputs a .py file that can be directly uploaded to Brain-Score.
Tests are directly generated here as well, and contained in here.
"""

from pathlib import Path


class MTurkBenchmarkFactory:
    def __call__(self):
        benchmark_code = self.generate_benchmark_code()
        # path = Path(benchmark_directory / plugin_name / 'benchmark.py')
        # write_code_into_file(benchmark_code, path)

    def generate_benchmark_code(self):
        benchmark_code = f"""
        class {plugin_name}Benchmark(BenchmarkBase):
            def __call__(self, candidate: BrainModel) -> Score: 
                ...
        """

    def generate_test_code(self) -> str:
        return f"""
        def test_num_trials(): ...
        """
