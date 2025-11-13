from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

from brainscore_core.supported_data_standards.brainio.fetch import get_stimulus_set
from brainscore_core.supported_data_standards.brainio.assemblies import NeuronRecordingAssembly
from brainscore_core.supported_data_standards.brainio.packaging import package_data_assembly


def main():

    stimuli = load_stimulus_set_from_s3(identifier="Rust2012",bucket="brainscore-storage/brainio-brainscore",
                                              csv_sha1="482da1f9f4a0ab5433c3b7b57073ad30e45c2bf1",
                                              zip_sha1="7cbf5dcec235f7705eaad1cfae202eda77e261a2",
                                              csv_version_id="null",
                                              zip_version_id="null"
                                            ).stimulus_set

    single_nc_path = Path("/Users/jjpr/dev/dldata/scripts/rust_single.nc")
    da_single = xr.open_dataarray(single_nc_path)
    da_single.name = 'dicarlo.Rust2012.single'

    print('Packaging assembly for single-unit')
    package_data_assembly(da_single, assembly_identifier=da_single.name, stimulus_set_identifier=stimuli.identifier,
                          bucket_name='brainio.dicarlo')

    array_nc_path = Path("/Users/jjpr/dev/dldata/scripts/rust_array.nc")
    da_array = xr.open_dataarray(array_nc_path)
    da_array.name = 'dicarlo.Rust2012.array'

    print('Packaging assembly for array')
    package_data_assembly(da_array, assembly_identifier=da_array.name, stimulus_set_identifier=stimuli.identifier,
                          bucket_name='brainio.dicarlo')


if __name__ == '__main__':
    main()


