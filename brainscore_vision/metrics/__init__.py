from brainscore_core.supported_data_standards.brainio.assemblies import DataAssembly
from brainscore_core import Score


class Ceiling:
    def __call__(self, assembly: DataAssembly) -> Score:
        raise NotImplementedError()
