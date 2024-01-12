from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

import brainio_collection
from brainio_base.assemblies import NeuronRecordingAssembly
from brainio_collection.packaging import package_data_assembly


def main():
    stimuli = brainio_collection.get_stimulus_set('dicarlo.Rust2012')

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


