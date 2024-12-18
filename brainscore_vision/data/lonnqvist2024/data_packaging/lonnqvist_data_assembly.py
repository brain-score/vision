from pathlib import Path
import numpy as np
import xarray as xr

from brainio.assemblies import BehavioralAssembly
from brainio.packaging import package_data_assembly
import pandas as pd


DATASETS = ['inlab-instructions', 'inlab-no-instructions', 'online-no-instructions']


def collect_lonnqvist_data_assembly(root_directory, dataset):
    """
    Experiment Information:
    """
    data = pd.read_csv(Path(rf'{root_directory}/{dataset}.csv'))

    assembly = BehavioralAssembly(data['subject_answer'],
                                  coords={
                                      'subject': ('presentation', data['subject_id']),
                                      'visual_degrees': ('presentation', data['visual_degrees']),
                                      'image_duration': ('presentation', data['image_duration']),
                                      'is_correct': ('presentation', data['is_correct']),
                                      'subject_answer': ('presentation', data['subject_answer']),
                                      'curve_length': ('presentation', data['curve_length']),
                                      'n_cross': ('presentation', data['n_cross']),
                                      'image_path': ('presentation', data['image_path']),
                                      'stimulus_id': ('presentation', data['stimulus_id']),
                                      'truth': ('presentation', data['truth']),
                                      'image_label': ('presentation', data['truth'])
                                  },
                                  dims=['presentation']
                                  )

    # give the assembly an identifier name
    assembly.name = f'Lonnqvist2024_{dataset}'

    return assembly


if __name__ == '__main__':
    root_directory = Path(r'./local')
    for dataset in DATASETS:
        assembly = collect_lonnqvist_data_assembly(root_directory, dataset)
        # upload to S3
        prints = package_data_assembly(catalog_identifier=None,
                                       proto_data_assembly=assembly,
                                       assembly_identifier=assembly.name,
                                       stimulus_set_identifier=assembly.name,
                                       assembly_class_name="BehavioralAssembly",
                                       bucket_name="brainscore-storage/brainio-brainscore")
        print(prints)
