
import numpy as np
import xarray
from brainio_collection.packaging import package_data_assembly
import scipy.io as sio
from brainio_base.assemblies import DataAssembly

DATA_DIR = '/braintree/data2/active/users/tmarques/bs_datasets/Ringach2002.mat'
ASSEMBLY_NAME = 'shapley.Ringach2002'
ORIENTATION_STIM_NAME = 'Marques2020_orientation'

PROPERTY_NAMES = ['baseline', 'max_dc', 'min_dc', 'max_ac', 'modulation_ratio', 'circular_variance', 'bandwidth',
                  'orthogonal_preferred_ratio', 'orientation_selective', 'circular_variance_bandwidth_ratio',
                  'orthogonal_preferred_ratio_circular_variance_difference',
                  'orthogonal_preferred_ratio_bandwidth_ratio']


def collect_data(data_dir):
    ringach2002 = sio.loadmat(data_dir)
    or_data = ringach2002['db']

    # Response magnitudes
    baseline = or_data['spont'][0, 0].T
    n_neuroids = baseline.shape[0]

    max_dc = or_data['maxdc'][0, 0].T
    min_dc = or_data['mindc'][0, 0].T
    max_ac = or_data['maxfirst'][0, 0].T
    modulation_ratio = max_ac / max_dc

    # Orientation tuning properties
    circular_variance = or_data['orivar'][0, 0].T
    bandwidth = or_data['bw'][0, 0].T
    bandwidth[bandwidth > 90] = np.nan
    orthogonal_preferred_ratio = or_data['po'][0, 0].T

    orientation_selective = np.ones((n_neuroids, 1))
    orientation_selective[np.isnan(bandwidth)] = 0

    # Orientation tuning properties covariances
    circular_variance_bandwidth_ratio = circular_variance / bandwidth
    orthogonal_preferred_ratio_circular_variance_difference = orthogonal_preferred_ratio - circular_variance
    orthogonal_preferred_ratio_bandwidth_ratio = orthogonal_preferred_ratio/bandwidth

    # Bins
    max_dc_bins = np.logspace(0, 3, 10, base=10)
    max_ac_bins = np.logspace(0, 3, 10, base=10)
    min_dc_bins = np.logspace(-1-1/3, 2, 11, base=10)
    baseline_bins = np.logspace(-1-1/3, 2, 11, base=10)
    modulation_ratio_bins = np.linspace(0, 2, 11)

    circular_variance_bins = np.linspace(0, 1, num=14)
    bandwidth_bins = np.linspace(0, 90, num=18)
    orthogonal_preferred_ratio_bins = np.linspace(0, 1, num=14)
    orientation_selective_bins = np.linspace(0, 1, num=3)

    circular_variance_bandwidth_ratio_bins = np.logspace(-3, 0, num=16, base=10)
    orthogonal_preferred_ratio_bandwidth_ratio_bins = np.logspace(-3, 0, num=16, base=10)
    orthogonal_preferred_ratio_circular_variance_difference_bins = np.linspace(-1, 1, num=20)

    # Create DataAssembly with single neuronal properties and bin information
    assembly = np.concatenate((baseline, max_dc, min_dc, max_ac, modulation_ratio, circular_variance, bandwidth,
                               orthogonal_preferred_ratio, orientation_selective,
                               circular_variance_bandwidth_ratio,
                               orthogonal_preferred_ratio_circular_variance_difference,
                               orthogonal_preferred_ratio_bandwidth_ratio), axis=1)

    # Filters neurons with weak responses
    good_neuroids = max_dc > baseline + 5
    assembly = assembly[np.argwhere(good_neuroids)[:, 0], :]

    assembly = DataAssembly(assembly, coords={'neuroid_id': ('neuroid', range(assembly.shape[0])),
                                              'region': ('neuroid', ['V1'] * assembly.shape[0]),
                                              'neuronal_property': PROPERTY_NAMES},
                            dims=['neuroid', 'neuronal_property'])

    assembly.attrs['number_of_trials'] = 20

    for p in assembly.coords['neuronal_property'].values:
        assembly.attrs[p+'_bins'] = eval(p+'_bins')

    return assembly


def main():
    assembly = collect_data(DATA_DIR)
    assembly.name = ASSEMBLY_NAME
    print('Packaging assembly')
    package_data_assembly(xarray.DataArray(assembly), assembly_identifier=assembly.name, stimulus_set_identifier=ORIENTATION_STIM_NAME,
                          assembly_class='PropertyAssembly', bucket_name='brainio.contriib')


if __name__ == '__main__':
    main()

