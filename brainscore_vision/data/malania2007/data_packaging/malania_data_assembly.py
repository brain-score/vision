from pathlib import Path
import numpy as np
import xarray as xr

from brainio.assemblies import PropertyAssembly
from brainio.packaging import package_data_assembly
import pandas as pd


DATASETS = ['short-2', 'short-4', 'short-6', 'short-8', 'short-16', 'equal-2',
            'long-2', 'equal-16', 'long-16', 'vernier-only']
NUM_SUBJECTS = {'short-2': 6,
                'short-4': 5,
                'short-6': 5,
                'short-8': 5,
                'short-16': 6,
                'equal-2': 5,
                'long-2': 5,
                'equal-16': 5,
                'long-16': 5,
                'vernier-only': 6}


def collect_malania_data_assembly(root_directory, dataset):
    """
    Experiment Information:
        - 5-6 observers per condition (for exact value, see NUM_SUBJECTS)
        - 2AFC left/right offset discrimination task
        - PEST staircase to 75% correct responses
        - thresholds measured with a cumulative gaussian psychometric function with a likelihood fit
    """
    # construct the assembly
    metadata_directory = Path(f'{root_directory}/{dataset}/metadata_human.xlsx')
    metadata = pd.read_excel(metadata_directory)
    # Since subjects are uniquely held using 'unique_subject_id', drop the rows with a subject
    #  without measurement
    assembly = PropertyAssembly(metadata['threshold'],
                                  coords={
                                      'subject_unique_id': ('subject', metadata['subject_unique_id'])
                                  },
                                  dims=['subject']
                                  )

    # give the assembly an identifier name
    assembly.name = f'Malania2007_{dataset}'

    # test subject numbers after removing the NaN subject
    metadata = metadata.dropna(subset=['threshold'], axis=0)
    assert len(metadata) == NUM_SUBJECTS[dataset]

    return assembly


def return_local_data_assembly(dataset):
    root_directory = Path(r'../../../packaging/malania2007/malania2007_data_assembly')
    assembly = collect_malania_data_assembly(root_directory, dataset)
    return assembly


def remove_subjects_with_nans(assembly1, assembly2):
    # Find the indices of the subjects with NaN values in the first PropertyAssembly
    nan_subjects = np.isnan(assembly1.values)

    # Convert the boolean array to a DataArray with the same coordinates as the input assemblies
    nan_subjects_da = xr.DataArray(nan_subjects, coords=assembly1.coords, dims=assembly1.dims)

    # Filter out the subjects with NaN values from both PropertyAssemblies
    filtered_assembly1 = assembly1.where(~nan_subjects_da, drop=True)
    filtered_assembly2 = assembly2.where(~nan_subjects_da, drop=True)

    return filtered_assembly1, filtered_assembly2


if __name__ == '__main__':
    root_directory = Path(r'../../data/malania2007/data_packaging/')
    for dataset in DATASETS:
        assembly = collect_malania_data_assembly(root_directory, dataset)
        # upload to S3
        prints = package_data_assembly(catalog_identifier=None,
                                       proto_data_assembly=assembly,
                                       assembly_identifier=assembly.name,
                                       stimulus_set_identifier=assembly.name,
                                       assembly_class_name="PropertyAssembly",
                                       bucket_name="brainio-brainscore")
        print(prints)