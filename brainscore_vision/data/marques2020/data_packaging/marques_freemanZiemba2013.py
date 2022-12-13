
import numpy as np
import xarray
from brainio_collection.packaging import package_data_assembly
import h5py
from brainio_base.assemblies import DataAssembly

DATA_DIR = '/braintree/data2/active/users/tmarques/bs_datasets/FreemanZiemba2013_V1V2data.mat'
ASSEMBLY_NAME = 'movshon.FreemanZiemba2013_V1_properties'
TEXTURE_STIM_NAME = 'movshon.FreemanZiemba2013_properties'

PROPERTY_NAMES = ['texture_modulation_index', 'absolute_texture_modulation_index', 'texture_selectivity',
                  'noise_selectivity', 'texture_sparseness', 'noise_sparseness', 'variance_ratio', 'sample_variance',
                  'family_variance', 'max_texture', 'max_noise', ]


def collect_data(data_dir):
    response_file = h5py.File(data_dir, 'r')

    texture_modulation_index, absolute_texture_modulation_index, texture_selectivity, noise_selectivity, \
    texture_sparseness, noise_sparseness, variance_ratio, sample_variance, family_variance, max_texture, max_noise = \
        calculate_texture_properties(response_file, area='v1')

    # Bins
    max_texture_bins = np.logspace(-1, 3, 13, base=10)
    max_noise_bins = np.logspace(-1, 3, 13, base=10)
    texture_modulation_index_bins = np.linspace(-1, 1, num=25)
    absolute_texture_modulation_index_bins = np.linspace(0, 1, num=13)
    texture_selectivity_bins = np.linspace(0, 1, num=11)
    noise_selectivity_bins = np.linspace(0, 1, num=11)
    texture_sparseness_bins = np.linspace(0, 1, num=11)
    noise_sparseness_bins = np.linspace(0, 1, num=11)
    variance_ratio_bins = np.logspace(-2, 1, num=13)
    sample_variance_bins = np.linspace(0, 1, num=11)
    family_variance_bins = np.linspace(0, 1, num=11)

    # Create DataAssembly with single neuronal properties and bin information
    assembly = np.concatenate((texture_modulation_index, absolute_texture_modulation_index, texture_selectivity,
                               noise_selectivity, texture_sparseness, noise_sparseness, variance_ratio, sample_variance,
                               family_variance, max_texture, max_noise), axis=1)

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
    package_data_assembly(xarray.DataArray(assembly), assembly_identifier=assembly.name, stimulus_set_identifier=TEXTURE_STIM_NAME,
                          assembly_class='PropertyAssembly', bucket_name='brainio.contrib')


def calculate_texture_properties(response_file, area='v1'):
    textureNumOrder = np.array([327, 336, 393, 402, 13, 18, 23, 30, 38, 48, 52, 56, 60, 71, 99])
    familyNumOrder = np.array([60, 56, 13, 48, 71, 18, 327, 336, 402, 38, 23, 52, 99, 393, 30])
    textureSorted = [np.argwhere(textureNumOrder == familyNumOrder[i])[0][0] for i in range(len(familyNumOrder))]

    responses = response_file[area].value
    # (cellNum) x (timeBin) x (rep) x (sample) x (texType) x (texFamily)
    # (102+)    x (300)     x (20)  x (15)     x (2)       x (15)

    responses = responses[:, :, :, :, :, textureSorted]
    responses = np.moveaxis(responses, 4, 2)  # texType to position 2
    responses = np.moveaxis(responses, 5, 3)  # family to position 3
    responses = np.moveaxis(responses, 5, 4)  # sample to position 4
    # (cellNum) x (timeBin) x (texType) x (texFamily) x (sample) x (rep)

    n_neurons, n_tb, n_type, n_fam, n_smp, n_rep = responses.shape

    responses_stabilized = np.copy(responses)  # copies data for treating as stabilized spike counts

    # Calculates average responses in 10ms timebins (in spike/s)
    responses = responses.reshape(n_neurons, 30, 10, n_type, n_fam, n_smp, n_rep).mean(axis=2) * 1000

    n_tb = responses.shape[1]
    sample_responses = responses.mean(axis=5)  # average across repetitions
    family_responses = sample_responses.mean(axis=4)  # average across samples
    type_responses = family_responses.mean(axis=3)  # average across types
    mean_responses = type_responses.mean(axis=2)  # average across all
    baseline = mean_responses.min(axis=1, keepdims=True)  # baseline response

    latency = calculate_latency(family_responses)  # gets latency using sample averages
    sample_responses = calculate_mean_response(sample_responses, lat=latency)  # mean response during 100ms
    sample_responses = sample_responses - baseline.reshape(-1, 1, 1, 1) # subtracts baseline response and rectifies
    sample_responses[sample_responses < 0] = 0
    family_responses = sample_responses.mean(axis=3)
    max_response = sample_responses.reshape((n_neurons, n_type, -1)).max(axis=2)

    # Sums spikes in 10ms timebins
    responses_stabilized = responses_stabilized.reshape(n_neurons, 30, 10, n_type, n_fam, n_smp, n_rep).sum(axis=2)
    responses_stabilized = calculate_mean_response(responses_stabilized, lat=latency) * 10
    responses_stabilized = np.sqrt(responses_stabilized) + np.sqrt(responses_stabilized + 1)
    responses_stabilized = responses_stabilized.mean(axis=4)

    # Maximum texture response
    max_texture = max_response[:, 1].reshape((-1, 1))

    # Maximum noise response
    max_noise = max_response[:, 0].reshape((-1, 1))

    texture_modulation_index = np.zeros((n_neurons, 1))
    texture_selectivity = np.zeros((n_neurons, 1))
    noise_selectivity = np.zeros((n_neurons, 1))
    texture_sparseness = np.zeros((n_neurons, 1))
    noise_sparseness = np.zeros((n_neurons, 1))
    variance_ratio = np.zeros((n_neurons, 1))
    sample_variance = np.zeros((n_neurons, 1))
    family_variance = np.zeros((n_neurons, 1))

    for n in range(n_neurons):
        # Texture modulation
        texture_modulation_index[n] = calculate_texture_modulation(family_responses[n])[0]
        # Texture selectivity (sparseness over texture families)
        texture_selectivity[n] = calculate_sparseness(family_responses[n, 1])
        # Noise selectivity (sparseness over noise families)
        noise_selectivity[n] = calculate_sparseness(family_responses[n, 0])
        # Texture sparseness (sparseness over texture samples)
        texture_sparseness[n] = calculate_sparseness(sample_responses[n, 1])
        # Noise sparseness (sparseness over noise samples)
        noise_sparseness[n] = calculate_sparseness(sample_responses[n, 0])
        variance_ratio[n], sample_variance[n], family_variance[n] = calculate_variance_ratio(responses_stabilized[n, 1])

    absolute_texture_modulation_index = np.abs(texture_modulation_index)

    return texture_modulation_index, absolute_texture_modulation_index, texture_selectivity, noise_selectivity,\
           texture_sparseness, noise_sparseness, variance_ratio, sample_variance, family_variance, max_texture, \
           max_noise


def calculate_mean_response(response, lat, resp_nbin=10):

    response_shape = response.shape
    n_neur = response_shape[0]
    mean_response = np.zeros(np.concatenate((np.array([n_neur]), np.array(response_shape[2:]))))

    for n in range(n_neur):
        mean_response[n] = response[n, lat[n]:(lat[n] + resp_nbin + 1)].mean(axis=0)

    return mean_response


def calculate_latency(response, t_pk=[7, 22], t_lat_min=2, max_d_lat=10, thrsh=0.15):

    # n_neur, n_tb, n_type, n_fam = meanByFam.shape

    response_shape = response.shape
    n_neur = response_shape[0]

    lat = np.zeros(n_neur).astype(int)
    for n in range(n_neur):
        pref_stim = np.unravel_index(np.argmax(response[n, t_pk[0]:t_pk[1]]), response_shape[1:])
        pk_lat = pref_stim[0]
        if len(pref_stim)==2:
            pref_response = response[n, :, pref_stim[1]]
        elif len(pref_stim)==3:
            pref_response = response[n, :, pref_stim[1], pref_stim[2]]
        elif len(pref_stim) == 4:
            pref_response = response[n, :, pref_stim[1], pref_stim[2], pref_stim[3]]

        pref_response_pk = np.max(pref_response)
        pk_lat = pk_lat + t_pk[0]

        min_lat = np.max([t_lat_min, pk_lat-max_d_lat])
        min_val = np.min(pref_response[min_lat:pk_lat])

        lat_min = np.argmin(pref_response[min_lat:pk_lat]) + min_lat
        range_val = pref_response_pk - min_val

        lat[n] = np.argwhere(pref_response[lat_min:(pk_lat + 1)] <= (min_val + thrsh * range_val))[-1][0] \
                 + lat_min + 1
    return lat


def calculate_texture_modulation(response):
    texMod_fam = (response[1, :] - response[0, :]) / (response[1, :] + response[0, :])
    texMod = np.nanmean(texMod_fam)

    return texMod, texMod_fam


def calculate_sparseness(response):

    response = response.reshape(-1)
    n_stim = response.shape[0]

    spars = (1 - ((response.sum() / n_stim) ** 2) / ((response ** 2).sum() / n_stim)) / (1 - 1 / n_stim)

    return spars


def calculate_variance_ratio(response):

    residual_ms, sample_ms, family_ms = calculate_variance(response)
    response_shape = response.shape

    if len(response_shape) == 3:
        residual_variance = residual_ms
        sample_variance = (sample_ms - residual_ms) / response_shape[2]
        family_variance = (family_ms - sample_ms) / (response_shape[2]*response_shape[1])

    else:
        residual_variance = 0
        sample_variance = sample_ms
        family_variance = (family_ms - sample_ms) / response_shape[1]

    total_variance = residual_variance + sample_variance + family_variance

    variance_ratio = (family_variance / total_variance + 0.02) / (sample_variance / total_variance + 0.02)

    return variance_ratio, sample_variance / total_variance, family_variance / total_variance


def calculate_variance(response):

    response_shape = response.shape

    if len(response_shape) == 3:
        a, b, n = response_shape

        sample_mean = response.mean(axis=2)
        family_mean = sample_mean.mean(axis=1)
        all_mean = family_mean.mean()

        residual_ms = np.sum((response - sample_mean.reshape(a, b, 1)) ** 2) / (a * b * (n - 1))
        sample_ms = n * np.sum((sample_mean - family_mean.reshape(a, 1)) ** 2) / (a * (b - 1))
        family_ms = b*n*np.sum((family_mean - all_mean) ** 2) / (a - 1)
    else:
        a, b = response_shape

        sample_mean = response
        family_mean = sample_mean.mean(axis=1)
        all_mean = family_mean.mean()

        residual_ms = np.nan
        sample_ms = np.sum((sample_mean - family_mean.reshape(a, 1)) ** 2) / (a * (b - 1))
        family_ms = b * np.sum((family_mean - all_mean) ** 2) / (a - 1)

    return residual_ms, sample_ms, family_ms


if __name__ == '__main__':
    main()
