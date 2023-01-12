import logging
import os
import sys

import xarray as xr

from brainio_base.assemblies import NeuroidAssembly
from brainio_contrib.packaging import package_data_assembly


def create_xarray(savepath):
    '''Packages the DataArray (stimulus set the same as HvM). Returns an xarray of ["neuroid", "presentation", "time_bin"]
    Note: using my "10ms" branch of dldata'''
    from dldata.stimulus_sets import hvm
    dataset = hvm.HvMWithDiscfade()
    assembly = dataset.xr_from_hvm_10ms_temporal()
    assembly.reset_index(assembly.indexes.keys(), inplace=True)
    assembly.to_netcdf(savepath)
    return assembly


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    for logger in ['peewee', 's3transfer', 'botocore', 'boto3', 'urllib3', 'PIL']:
        logging.getLogger(logger).setLevel(logging.INFO)

    assembly_path = os.path.join(os.path.dirname(__file__), 'darren_xr.nc')
    create_xarray(assembly_path)  # Note: this function was run separately by @anayebi
    assembly = xr.open_dataarray(assembly_path)
    assembly = NeuroidAssembly(assembly)
    package_data_assembly(assembly, data_assembly_name='dicarlo.Majaj2015.temporal-10ms', bucket_name='brainio-dicarlo',
                          stimulus_set_name='dicarlo.hvm')
