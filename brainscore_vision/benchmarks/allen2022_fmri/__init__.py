from brainscore_vision import benchmark_registry
from .benchmark import Allen2022fmri, Allen2022fmriRSA

# Default: 8 subjects, 515 images
benchmark_registry['Allen2022_fmri.V1-ridge'] = lambda: Allen2022fmri('V1', 'ridge')
benchmark_registry['Allen2022_fmri.V2-ridge'] = lambda: Allen2022fmri('V2', 'ridge')
benchmark_registry['Allen2022_fmri.V4-ridge'] = lambda: Allen2022fmri('V4', 'ridge')
benchmark_registry['Allen2022_fmri.IT-ridge'] = lambda: Allen2022fmri('IT', 'ridge')

benchmark_registry['Allen2022_fmri.V1-rdm'] = lambda: Allen2022fmriRSA('V1')
benchmark_registry['Allen2022_fmri.V2-rdm'] = lambda: Allen2022fmriRSA('V2')
benchmark_registry['Allen2022_fmri.V4-rdm'] = lambda: Allen2022fmriRSA('V4')
benchmark_registry['Allen2022_fmri.IT-rdm'] = lambda: Allen2022fmriRSA('IT')

# 4-subject variant: subjects 1,2,5,7 (~1000 images)
benchmark_registry['Allen2022_fmri_4subj.V1-ridge'] = lambda: Allen2022fmri('V1', 'ridge', dataset_prefix='Allen2022_fmri_4subj')
benchmark_registry['Allen2022_fmri_4subj.V2-ridge'] = lambda: Allen2022fmri('V2', 'ridge', dataset_prefix='Allen2022_fmri_4subj')
benchmark_registry['Allen2022_fmri_4subj.V4-ridge'] = lambda: Allen2022fmri('V4', 'ridge', dataset_prefix='Allen2022_fmri_4subj')
benchmark_registry['Allen2022_fmri_4subj.IT-ridge'] = lambda: Allen2022fmri('IT', 'ridge', dataset_prefix='Allen2022_fmri_4subj')

benchmark_registry['Allen2022_fmri_4subj.V1-rdm'] = lambda: Allen2022fmriRSA('V1', dataset_prefix='Allen2022_fmri_4subj')
benchmark_registry['Allen2022_fmri_4subj.V2-rdm'] = lambda: Allen2022fmriRSA('V2', dataset_prefix='Allen2022_fmri_4subj')
benchmark_registry['Allen2022_fmri_4subj.V4-rdm'] = lambda: Allen2022fmriRSA('V4', dataset_prefix='Allen2022_fmri_4subj')
benchmark_registry['Allen2022_fmri_4subj.IT-rdm'] = lambda: Allen2022fmriRSA('IT', dataset_prefix='Allen2022_fmri_4subj')
