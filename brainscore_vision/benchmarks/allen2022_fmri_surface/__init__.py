from brainscore_vision import benchmark_registry
from .benchmark import Allen2022fmriSurface, Allen2022fmriSurfaceRSA

# Default: 8 subjects, 515 images
benchmark_registry['Allen2022_fmri_surface.V1-ridge'] = lambda: Allen2022fmriSurface('V1', 'ridge')
benchmark_registry['Allen2022_fmri_surface.V2-ridge'] = lambda: Allen2022fmriSurface('V2', 'ridge')
benchmark_registry['Allen2022_fmri_surface.V4-ridge'] = lambda: Allen2022fmriSurface('V4', 'ridge')
benchmark_registry['Allen2022_fmri_surface.IT-ridge'] = lambda: Allen2022fmriSurface('IT', 'ridge')

benchmark_registry['Allen2022_fmri_surface.V1-rdm'] = lambda: Allen2022fmriSurfaceRSA('V1')
benchmark_registry['Allen2022_fmri_surface.V2-rdm'] = lambda: Allen2022fmriSurfaceRSA('V2')
benchmark_registry['Allen2022_fmri_surface.V4-rdm'] = lambda: Allen2022fmriSurfaceRSA('V4')
benchmark_registry['Allen2022_fmri_surface.IT-rdm'] = lambda: Allen2022fmriSurfaceRSA('IT')

# 4-subject variant: subjects 1,2,5,7 (~1000 images)
benchmark_registry['Allen2022_fmri_surface_4subj.V1-ridge'] = lambda: Allen2022fmriSurface('V1', 'ridge', dataset_prefix='Allen2022_fmri_surface_4subj')
benchmark_registry['Allen2022_fmri_surface_4subj.V2-ridge'] = lambda: Allen2022fmriSurface('V2', 'ridge', dataset_prefix='Allen2022_fmri_surface_4subj')
benchmark_registry['Allen2022_fmri_surface_4subj.V4-ridge'] = lambda: Allen2022fmriSurface('V4', 'ridge', dataset_prefix='Allen2022_fmri_surface_4subj')
benchmark_registry['Allen2022_fmri_surface_4subj.IT-ridge'] = lambda: Allen2022fmriSurface('IT', 'ridge', dataset_prefix='Allen2022_fmri_surface_4subj')

benchmark_registry['Allen2022_fmri_surface_4subj.V1-rdm'] = lambda: Allen2022fmriSurfaceRSA('V1', dataset_prefix='Allen2022_fmri_surface_4subj')
benchmark_registry['Allen2022_fmri_surface_4subj.V2-rdm'] = lambda: Allen2022fmriSurfaceRSA('V2', dataset_prefix='Allen2022_fmri_surface_4subj')
benchmark_registry['Allen2022_fmri_surface_4subj.V4-rdm'] = lambda: Allen2022fmriSurfaceRSA('V4', dataset_prefix='Allen2022_fmri_surface_4subj')
benchmark_registry['Allen2022_fmri_surface_4subj.IT-rdm'] = lambda: Allen2022fmriSurfaceRSA('IT', dataset_prefix='Allen2022_fmri_surface_4subj')
