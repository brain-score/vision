from brainscore_vision import benchmark_registry
from .benchmark import Allen2022fmriSurface, Allen2022fmriSurfaceRSA

benchmark_registry['Allen2022_fmri_surface.V1-ridge'] = lambda: Allen2022fmriSurface('V1', 'ridge')
benchmark_registry['Allen2022_fmri_surface.V2-ridge'] = lambda: Allen2022fmriSurface('V2', 'ridge')
benchmark_registry['Allen2022_fmri_surface.V4-ridge'] = lambda: Allen2022fmriSurface('V4', 'ridge')
benchmark_registry['Allen2022_fmri_surface.IT-ridge'] = lambda: Allen2022fmriSurface('IT', 'ridge')

benchmark_registry['Allen2022_fmri_surface.V1-rdm'] = lambda: Allen2022fmriSurfaceRSA('V1')
benchmark_registry['Allen2022_fmri_surface.V2-rdm'] = lambda: Allen2022fmriSurfaceRSA('V2')
benchmark_registry['Allen2022_fmri_surface.V4-rdm'] = lambda: Allen2022fmriSurfaceRSA('V4')
benchmark_registry['Allen2022_fmri_surface.IT-rdm'] = lambda: Allen2022fmriSurfaceRSA('IT')
