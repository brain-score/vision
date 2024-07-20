
import numpy as np
import xarray
from marques_utils import gen_sample
from brainio_collection.packaging import package_data_assembly
from brainio_base.assemblies import DataAssembly

ASSEMBLY_NAME = 'schiller.Schiller1976c'
SPATIAL_FREQUENCY_STIM_NAME = 'Marques2020_spatial_frequency'


def collect_data():
    n_neurons = 87
    # Spatial frequency selective data
    spatial_frequency_selective_bins = np.linspace(0, 1, num=3)
    spatial_frequency_selective_hist = np.array([14, 73])
    spatial_frequency_selective = np.concatenate((np.zeros(spatial_frequency_selective_hist[0]),
                                                  np.ones(spatial_frequency_selective_hist[1]))).reshape((-1,1))

    # Spatial frequency bandwidth data
    spatial_frequency_bandwidth_bins = np.linspace(0, 100, num=11)
    spatial_frequency_bandwidth_simple_hist = np.array([0, 0, 0, 6, 5, 10, 7, 2, 0, 0])
    spatial_frequency_bandwidth_complex_hist = np.array([0, 3, 4, 10, 17, 6, 2, 1, 0, 0])
    spatial_frequency_bandwidth_hist = (spatial_frequency_bandwidth_simple_hist +
                                        spatial_frequency_bandwidth_complex_hist)
    spatial_frequency_bandwidth = gen_sample(spatial_frequency_bandwidth_hist, spatial_frequency_bandwidth_bins,
                                             scale='linear')
    filler = np.zeros((n_neurons - spatial_frequency_bandwidth_hist.sum(), 1))
    filler[:] = np.nan
    spatial_frequency_bandwidth = np.concatenate((filler, spatial_frequency_bandwidth))

    # Create DataAssembly with single neuronal properties and bin information
    assembly = np.concatenate((spatial_frequency_selective, spatial_frequency_bandwidth), axis=1)

    assembly = DataAssembly(assembly, coords={'neuroid_id': ('neuroid', range(assembly.shape[0])),
                                              'region': ('neuroid', ['V1'] * assembly.shape[0]),
                                              'neuronal_property': ['spatial_frequency_selective',
                                                                    'spatial_frequency_bandwidth']},
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

