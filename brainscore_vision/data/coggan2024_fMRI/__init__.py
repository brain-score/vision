# Created by David Coggan on 2024 06 23

from brainio.assemblies import NeuroidAssembly
from brainscore_vision import (
    stimulus_set_registry, data_registry, load_stimulus_set)
from brainscore_vision.data_helpers.s3 import (
    load_assembly_from_s3, load_stimulus_set_from_s3)

# stimulus set
stimulus_set_registry['Coggan2024_fMRI'] = lambda: load_stimulus_set_from_s3(
    identifier="tong.Coggan2024_fMRI",
    bucket="brainio-brainscore",
    csv_sha1="0089f5f8fd3f2de14de12ed736a0f88575f8e1ee",
    zip_sha1="e26fdea4d866799526dea183f5bfb9792718822a",
    csv_version_id="q0kxLCC8m6LSaLEQzB26sfyoKgJJwkAs",
    zip_version_id="SZwyN4ZgEmpW22YlcbqIbL9c0QudlMAp")

# fMRI data
data_registry['Coggan2024_fMRI'] = lambda: load_assembly_from_s3(
    identifier="tong.Coggan2024_fMRI",
    version_id="6P898Mio3VBsFx_qbC9rJECMELNGRprH",
    sha1="da3adbca5247d0491d366f94e8431fb3e4e58db2",
    bucket="brainio-brainscore",
    cls=NeuroidAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Coggan2024_fMRI'),
)

