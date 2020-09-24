import brainscore
from brainscore.benchmarks._properties_common import PropertiesBenchmark
from brainscore.metrics.ceiling import NeuronalPropertyCeiling
from brainio_base.assemblies import DataAssembly
from brainscore.metrics.distribution_similarity import BootstrapDistributionSimilarity, ks_sim
from brainscore.utils.v1_properties_utils import calc_cv, calc_bw, calc_opr
from result_caching import store
import numpy as np

ASSEMBLY_NAME = 'dicarlo.Marques2020_Ringach2002'
REGION = 'V1'
PARENT_ORIENTATION = 'V1-orientation'
PARENT_MAGNITUDE = 'V1-magnitude'

PROPERTY_NAMES = ['baseline', 'maxdc', 'mindc', 'maxac', 'mod_ratio', 'circ_var', 'bandwidth', 'orth_pref_ratio',
                  'or_sel', 'cv_bw_ratio', 'opr_cv_diff', 'opr_bw_ratio']

BIBTEX = """@article{Ringach2002,
            abstract = {We studied the steady-state orientation selectivity of single neurons in macaque primary visual cortex (V1). To analyze the data, two measures of orientation tuning selectivity, circular variance and orientation bandwidth, were computed from the tuning curves. Circular variance is a global measure of the shape of the tuning curve, whereas orientation bandwidth is a local measure of the sharpness of the tuning curve around its peak. Circular variance in V1 was distributed broadly, indicating a great diversity of orientation selectivity. This diversity was also reflected in the individual cortical layers. However, there was a tendency for neurons with high circular variance, meaning low selectivity for orientation, to be concentrated in layers 4C, 3B, and 5. The relative variation of orientation bandwidth across the cortical layers was less than for circular variance, but it showed a similar laminar dependence. Neurons with large orientation bandwidth were found predominantly in layers 4C and 3B. There was a weak correlation between orientation selectivity and the level of spontaneous activity of the neurons. We also assigned a response modulation ratio for each cell, which is a measure of the linearity of spatial summation. Cells with low modulation ratios tended to have higher circular variance and bandwidth than those with high modulation ratios. These findings suggest a revision to the classical view that nonoriented receptive fields are principally found in layer 4C and the cytochrome oxidase-rich blobs in layer 2/3. Instead, a broad distribution of tuning selectivity is found in all cortical layers, and neurons that are weakly tuned for orientation are ubiquitous in V1 cortex.},
            author = {Ringach, Dario L and Shapley, Robert M and Hawken, Michael J},
            doi = {20026567},
            file = {:Users/tiago/Documents/Mendeley Desktop/Ringach, Shapley, Hawken/Ringach, Shapley, Hawken{\_}2002.pdf:pdf},
            issn = {1529-2401},
            journal = {The Journal of Neuroscience},
            keywords = {1981,1998,bandwidth,circular variance,cortical layer,findings is that there,great diversity of circular,is a,ity,one of our main,orientation selectiv-,primate vision,striate cortex,swindale,variance in v1},
            number = {13},
            pages = {5639--5651},
            pmid = {12097515},
            title = {{Orientation selectivity in macaque V1: diversity and laminar dependence.}},
            url = {http://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed{\&}id=12097515{\&}retmode=ref{\&}cmd=prlinks{\%}5Cnpapers3://publication/uuid/EA0B2785-11B2-41CE-AF09-71647A4E026D},
            volume = {22},
            year = {2002}
            }"""


def _MarquesRingach2002V1Property(property_name, similarity_metric, ceil_func, parent):
    assembly = brainscore.get_assembly(ASSEMBLY_NAME)
    return PropertiesBenchmark(identifier=f'dicarlo.Marques_ringach2002.V1-{property_name}', assembly=assembly,
                               neuronal_property=ringach2002_properties, similarity_metric=similarity_metric,
                               parent=parent, ceiling_func=ceil_func, bibtex=BIBTEX, version=0.1)


def MarquesRingach2002V1CV():
    property_name = 'circ_var'
    parent = PARENT_ORIENTATION
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_sim, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(BootstrapDistributionSimilarity(similarity_func=ks_sim,
                                                                        property_name=property_name))
    return _MarquesRingach2002V1Property(property_name=property_name, similarity_metric=similarity_metric,
                                         ceil_func=ceil_func, parent=parent)


def MarquesRingach2002V1BW():
    property_name = 'bandwidth'
    parent = PARENT_ORIENTATION
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_sim, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(BootstrapDistributionSimilarity(similarity_func=ks_sim,
                                                                        property_name=property_name))
    return _MarquesRingach2002V1Property(property_name=property_name, similarity_metric=similarity_metric,
                                         ceil_func=ceil_func, parent=parent)


def MarquesRingach2002V1OPR():
    property_name = 'orth_pref_ratio'
    parent = PARENT_ORIENTATION
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_sim, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(BootstrapDistributionSimilarity(similarity_func=ks_sim,
                                                                        property_name=property_name))
    return _MarquesRingach2002V1Property(property_name=property_name, similarity_metric=similarity_metric,
                                         ceil_func=ceil_func, parent=parent)

def MarquesRingach2002V1OS():
    property_name = 'or_sel'
    parent = PARENT_ORIENTATION
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_sim, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(BootstrapDistributionSimilarity(similarity_func=ks_sim,
                                                                        property_name=property_name))
    return _MarquesRingach2002V1Property(property_name=property_name, similarity_metric=similarity_metric,
                                         ceil_func=ceil_func, parent=parent)


def MarquesRingach2002V1MAXDC():
    property_name = 'maxdc'
    parent = PARENT_MAGNITUDE
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_sim, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(BootstrapDistributionSimilarity(similarity_func=ks_sim,
                                                                        property_name=property_name))
    return _MarquesRingach2002V1Property(property_name=property_name, similarity_metric=similarity_metric,
                                         ceil_func=ceil_func, parent=parent)


def MarquesRingach2002V1MAXAC():
    property_name = 'maxac'
    parent = PARENT_MAGNITUDE
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_sim, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(BootstrapDistributionSimilarity(similarity_func=ks_sim,
                                                                        property_name=property_name))
    return _MarquesRingach2002V1Property(property_name=property_name, similarity_metric=similarity_metric,
                                         ceil_func=ceil_func, parent=parent)


def MarquesRingach2002V1MR():
    property_name = 'mod_ratio'
    parent = PARENT_MAGNITUDE
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_sim, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(BootstrapDistributionSimilarity(similarity_func=ks_sim,
                                                                        property_name=property_name))
    return _MarquesRingach2002V1Property(property_name=property_name, similarity_metric=similarity_metric,
                                         ceil_func=ceil_func, parent=parent)


def MarquesRingach2002V1BSLN():
    property_name = 'baseline'
    parent = PARENT_MAGNITUDE
    similarity_metric = BootstrapDistributionSimilarity(similarity_func=ks_sim, property_name=property_name)
    ceil_func = NeuronalPropertyCeiling(BootstrapDistributionSimilarity(similarity_func=ks_sim,
                                                                        property_name=property_name))
    return _MarquesRingach2002V1Property(property_name=property_name, similarity_metric=similarity_metric,
                                         ceil_func=ceil_func, parent=parent)


@store(identifier_ignore=['responses', 'baseline'])
def ringach2002_properties(model_identifier, responses, baseline):
    radius = np.array(sorted(set(responses.radius.values)))
    sf = np.array(sorted(set(responses.sf.values)))
    orientation = np.array(sorted(set(responses.orientation.values)))
    phase = np.array(sorted(set(responses.phase.values)))

    responses = responses.values
    baseline = baseline.values
    assert responses.shape[0] == baseline.shape[0]
    n_neuroids = responses.shape[0]

    responses = responses.reshape((n_neuroids, len(radius), len(sf), len(orientation), len(phase)))
    responses = np.concatenate((responses[:, 0:1, 2, :, :], responses[:, 1:2, 1, :, :], responses[:, 2:, 0, :, :]),
                               axis=1)
    responses_dc = responses.mean(axis=3)
    responses_fft = np.absolute(np.fft.fft(responses)) / len(phase)
    responses_ac = responses_fft[:, :, :, 1]

    maxdc = np.zeros((n_neuroids, 1))
    maxac = np.zeros((n_neuroids, 1))
    mindc = np.zeros((n_neuroids, 1))
    circ_var = np.zeros((n_neuroids, 1))
    bandwidth = np.zeros((n_neuroids, 1))
    orth_pref_ratio = np.zeros((n_neuroids, 1))
    or_sel = np.ones((n_neuroids, 1))

    for neur in range(n_neuroids):
        pref_sf, pref_orientation = np.unravel_index(np.argmax(responses_dc[neur, :, :]), (len(sf), len(orientation)))

        maxdc[neur] = responses_dc[neur, pref_sf, pref_orientation]
        maxac[neur] = responses_ac[neur, pref_sf, pref_orientation]

        orientation_curve = responses_dc[neur, pref_sf]
        mindc[neur] = orientation_curve.min()

        circ_var[neur] = calc_cv(orientation_curve, orientation)
        bandwidth[neur],  pref_or_fit, or_full, or_curve_full = \
            calc_bw(orientation_curve, orientation, filt_type='hanning', thrsh=0.707, mode='half')
        orth_pref_ratio[neur] = calc_opr(orientation_curve, orientation)

    or_sel[np.isnan(bandwidth)] = 0
    mod_ratio = maxac / maxdc
    cv_bw_ratio = circ_var / bandwidth
    opr_cv_diff = orth_pref_ratio - circ_var
    opr_bw_ratio = orth_pref_ratio/bandwidth

    properties_data = np.concatenate((baseline, maxdc, mindc, maxac, mod_ratio, circ_var, bandwidth, orth_pref_ratio,
                                      or_sel, cv_bw_ratio, opr_cv_diff, opr_bw_ratio), axis=1)
    good_neuroids = maxdc > baseline + 5
    properties_data = properties_data[np.argwhere(good_neuroids)[:, 0], :]

    properties_data = DataAssembly(properties_data, coords={'neuroid_id': ('neuroid', range(properties_data.shape[0])),
                                                            'region': ('neuroid', ['V1'] * properties_data.shape[0]),
                                                            'neuronal_property': PROPERTY_NAMES},
                                   dims=['neuroid', 'neuronal_property'])
    return properties_data

