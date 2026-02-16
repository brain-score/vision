from brainscore_vision import benchmark_registry
from .benchmark import Allen2022fmri, Allen2022fmriRSA

benchmark_registry['Allen2022_fmri.V1-ridge'] = lambda: Allen2022fmri('V1', 'ridge')
benchmark_registry['Allen2022_fmri.V2-ridge'] = lambda: Allen2022fmri('V2', 'ridge')
benchmark_registry['Allen2022_fmri.V4-ridge'] = lambda: Allen2022fmri('V4', 'ridge')
benchmark_registry['Allen2022_fmri.IT-ridge'] = lambda: Allen2022fmri('IT', 'ridge')

benchmark_registry['Allen2022_fmri.V1-rdm'] = lambda: Allen2022fmriRSA('V1')
benchmark_registry['Allen2022_fmri.V2-rdm'] = lambda: Allen2022fmriRSA('V2')
benchmark_registry['Allen2022_fmri.V4-rdm'] = lambda: Allen2022fmriRSA('V4')
benchmark_registry['Allen2022_fmri.IT-rdm'] = lambda: Allen2022fmriRSA('IT')
