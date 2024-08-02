
import numpy as np
import xarray
from marques_utils import gen_sample
from brainio_collection.packaging import package_data_assembly
from brainio_base.assemblies import DataAssembly

ASSEMBLY_NAME = 'movshon.Cavanaugh2002a'
SIZE_STIM_NAME = 'Marques2020_size'

PROPERTY_NAMES = ['surround_suppression_index', 'strongly_suppressed', 'grating_summation_field', 'surround_diameter',
                  'surround_grating_summation_field_ratio']


def collect_data():
    n_neurons = 190

    # Surround suppression index (n=190 neurons, foveal eccentricity, response > 5spk/s)
    surround_suppression_index_bins = np.linspace(0, 2, num=11)
    surround_suppression_index_hist = np.array([66, 44, 38, 23, 16, 3, 0, 0, 0, 0])
    surround_suppression_index = gen_sample(surround_suppression_index_hist, surround_suppression_index_bins,
                                            scale='linear')

    # Strongly suppressed (n=190 neurons, foveal eccentricity, response > 5spk/s)
    strongly_suppressed_bins = np.linspace(0, 1, num=3)
    strongly_suppressed_hist = np.array([42, 148])
    strongly_suppressed = np.concatenate((np.zeros(strongly_suppressed_hist[0]),
                                          np.ones(strongly_suppressed_hist[1]))).reshape((-1, 1))

    # Grating summation field (n=148 neurons & suppression > 10%)
    grating_summation_field_bins = np.array([0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    grating_summation_field_hist = np.array([0, 9, 30, 55, 42, 10, 2, 0])
    grating_summation_field = gen_sample(grating_summation_field_hist, grating_summation_field_bins, scale='log2')
    filler = np.zeros((n_neurons - grating_summation_field_hist.sum(), 1))
    filler[:] = np.nan
    grating_summation_field = np.concatenate((filler, grating_summation_field))

    # Surround diameter (n=148 neurons & suppression > 10%)
    surround_diameter_bins = np.array([0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    surround_diameter_hist = np.array([0, 0, 2, 20, 33, 49, 40, 4])
    surround_diameter = gen_sample(surround_diameter_hist, surround_diameter_bins, scale='log2')
    filler = np.zeros((n_neurons - surround_diameter_hist.sum(), 1))
    filler[:] = np.nan
    surround_diameter = np.concatenate((filler, surround_diameter))

    # Surround diameter / Grating summation field (n=148 neurons, foveal eccentricity & suppression > 10%)
    surround_grating_summation_field_ratio_bins = np.array([1, 2, 4, 8, 16, 32])
    surround_grating_summation_field_ratio_hist = np.array([69, 38, 31, 10, 0])
    surround_grating_summation_field_ratio = gen_sample(surround_grating_summation_field_ratio_hist,
                                                        surround_grating_summation_field_ratio_bins, scale='log2')
    filler = np.zeros((n_neurons - surround_grating_summation_field_ratio_hist.sum(), 1))
    filler[:] = np.nan
    surround_grating_summation_field_ratio = np.concatenate((filler, surround_grating_summation_field_ratio))

    # Create DataAssembly with single neuronal properties and bin information
    assembly = np.concatenate((surround_suppression_index, strongly_suppressed, grating_summation_field,
                               surround_diameter, surround_grating_summation_field_ratio), axis=1)

    assembly = DataAssembly(assembly, coords={'neuroid_id': ('neuroid', range(assembly.shape[0])),
                                              'region': ('neuroid', ['V1'] * assembly.shape[0]),
                                              'neuronal_property': PROPERTY_NAMES},
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
                          stimulus_set_identifier=SIZE_STIM_NAME, assembly_class='PropertyAssembly',
                          bucket_name='brainio.contrib')


if __name__ == '__main__':
    main()

