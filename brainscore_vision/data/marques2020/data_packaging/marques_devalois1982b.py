
import numpy as np
import xarray
from marques_utils import gen_sample
from brainio_collection.packaging import package_data_assembly
from brainio_base.assemblies import DataAssembly

ASSEMBLY_NAME = 'devalois.DeValois1982b'
SPATIAL_FREQUENCY_STIM_NAME = 'Marques2020_spatial_frequency'


def collect_data():
    # Peak spatial frequency data
    peak_spatial_frequency_bins = np.array([0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.2, 16.0, 22.4])

    peak_spatial_frequency_simple_foveal_hist = np.array([0, 4, 4, 8, 25, 33, 26, 28, 12, 5, 2, 1])
    peak_spatial_frequency_complex_foveal_hist = np.array([0, 0, 0, 9, 9, 7, 10, 23, 12, 8, 3, 3])
    peak_spatial_frequency_simple_parafoveal_hist = np.array([2, 4, 10, 12, 18, 7, 18, 3, 4, 0, 0, 0])
    peak_spatial_frequency_complex_parafoveal_hist = np.array([1, 2, 1, 2, 5, 15, 13, 9, 3, 2, 0, 0])

    peak_spatial_frequency_hist = (
                peak_spatial_frequency_simple_foveal_hist + peak_spatial_frequency_complex_foveal_hist +
                peak_spatial_frequency_simple_parafoveal_hist + peak_spatial_frequency_complex_parafoveal_hist)

    peak_spatial_frequency = gen_sample(peak_spatial_frequency_hist, peak_spatial_frequency_bins, scale='log2')

    # Create DataAssembly with single neuronal properties and bin information
    assembly = DataAssembly(peak_spatial_frequency, coords={'neuroid_id': ('neuroid',
                                                                           range(peak_spatial_frequency.shape[0])),
                                                            'region': ('neuroid', ['V1'] *
                                                                       peak_spatial_frequency.shape[0]),
                                                            'neuronal_property': ['peak_spatial_frequency']},
                            dims=['neuroid', 'neuronal_property'])

    assembly.attrs['number_of_trials'] = 20

    for p in assembly.coords['neuronal_property'].values:
        assembly.attrs[p+'_bins'] = eval(p+'_bins')

    return assembly


def main():
    assembly = collect_data()
    assembly.name = ASSEMBLY_NAME
    print('Packaging assembly')
    package_data_assembly(xarray.DataArray(assembly), assembly_identifier=assembly.name,
                          stimulus_set_identifier=SPATIAL_FREQUENCY_STIM_NAME, assembly_class='PropertyAssembly',
                          bucket_name='brainio.contrib')


if __name__ == '__main__':
    main()

