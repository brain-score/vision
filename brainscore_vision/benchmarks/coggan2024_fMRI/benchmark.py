# Created by David Coggan on 2024 06 25

import numpy as np
from brainio.assemblies import DataAssembly, NeuroidAssembly
from brainscore_vision import load_dataset
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_core.metrics import Score
from brainscore_vision.metric_helpers import Defaults as XarrayDefaults
from brainscore_vision.model_interface import BrainModel


# the BIBTEX will be used to link to the publication from the benchmark for further details
BIBTEX = """@article {
    Tong.Coggan2024.fMRI,
    author = {David D. Coggan and Frank Tong},
    title = {Evidence of strong amodal completion in both early and 
    high-level visual cortices},
    year = {2024},
    url = {},
    journal = {under review}}"""


class Coggan2024_fMRI_Benchmark(BenchmarkBase):
    def __init__(self, identifier, assembly, ceiling_func, visual_degrees,
                 **kwargs):
        super(Coggan2024_fMRI_Benchmark, self).__init__(
            identifier=identifier,
            ceiling_func=ceiling_func, **kwargs)
        self._assembly = assembly
        self._visual_degrees = visual_degrees
        self._ceiling_func = ceiling_func
        region = np.unique(self._assembly['region'])
        assert len(region) == 1
        self.region = region[0]

    def __call__(self, candidate: BrainModel):

        # get stimuli
        stimulus_set = place_on_screen(
            self._assembly.stimulus_set,
            target_visual_degrees=candidate.visual_degrees(),
            source_visual_degrees=self._visual_degrees)

        # get model activations
        candidate.start_recording(self.region, time_bins=[(0, 250)])
        source_assembly = candidate.look_at(stimulus_set, number_of_trials=1)
        if ('time_bin' in source_assembly.dims and
                source_assembly.sizes['time_bin'] == 1):
            source_assembly = source_assembly.squeeze('time_bin')

        # obtain the RSM
        source_rsm = RSA(source_assembly, XarrayDefaults.neuroid_dim)

        # compare the model and human RSMs to get a score
        raw_score = get_score(source_rsm, self._assembly)

        # obtain the score ceiling
        ceiling = self._ceiling_func(self._assembly)

        # obtain the ceiled score
        ceiled_score = ceiler(raw_score, ceiling)

        return ceiled_score


def get_ceiling(assembly: NeuroidAssembly) -> Score:

    """
    Returns the noise ceiling for the roi of the assembly.
    This is the lower bound of typical noise-ceiling range
    (e.g. Nili et al., 2014), i.e., the correlation of each individual
    subject's RSM with the mean RSM across the remaining subjects in the sample.
    This matches how the model is scored, if the group RSM is substituted for
    model RSM.
    """

    off_diag_indices = np.array(1 - np.eye(24).flatten(), dtype=bool)
    assert len(set(assembly.region.values)) == 1
    nc = []
    n_subs = len(assembly['subject'])
    for s in range(n_subs):
        # get individual and group RSMs, flatten, remove on-diagonal vals
        RSM_ind = assembly.values[:, :, s].flatten()[off_diag_indices]
        RSM_grp = assembly.values[:, :, [i for i in range(n_subs) if i != s]
                  ].mean(axis=2).flatten()[off_diag_indices]
        nc.append(np.corrcoef(RSM_ind, RSM_grp)[0, 1])
    noise_ceiling = Score(np.mean(nc))
    noise_ceiling.attrs['raw'] = nc
    return noise_ceiling


def get_score(source_rsm: NeuroidAssembly, target_rsm: NeuroidAssembly) -> Score:

    """
    Computes the pearson correlation between the model RSM and each subject's
    RSM, the average of which is returned as the score. Individual scores are
    stored in the Score's attributes.
    """

    off_diag_indices = np.array(1 - np.eye(24).flatten(), dtype=bool)
    model_rsm = source_rsm.values.flatten()[off_diag_indices]
    scores = []
    n_subs = len(target_rsm['subject'])
    for s in range(n_subs):
        human_rsm = target_rsm.values[:, :, s].flatten()[off_diag_indices]
        scores.append(np.corrcoef(human_rsm, model_rsm)[0, 1])
    score = Score(np.mean(scores))
    score.attrs[Score.RAW_VALUES_KEY] = scores
    return score


def ceiler(score: Score, ceiling: Score) -> Score:
    # ro(X, Y)
    # = (r(X, Y) / sqrt(r(X, X) * r(Y, Y)))^2
    # = (r(X, Y) / sqrt(r(Y, Y) * r(Y, Y)))^2  # assuming that r(Y, Y) ~ r(X, X) following Yamins 2014
    # = (r(X, Y) / r(Y, Y))^2
    r_square = np.power(score.values / ceiling.values, 2)
    ceiled_score = Score(r_square)
    if 'error' in score.attrs:
        ceiled_score.attrs['error'] = score.attrs['error']
    ceiled_score.attrs[Score.RAW_VALUES_KEY] = score
    ceiled_score.attrs['ceiling'] = ceiling
    return ceiled_score


def RSA(assembly: NeuroidAssembly, neuroid_dim: str) -> DataAssembly:

    """
    Performs analogous unit selection and normalization as the fMRI analysis,
    then calculates RSMs.
    """

    # get data orientation
    assert neuroid_dim in assembly.dims, \
        f'neuroid_dim {neuroid_dim} not in assembly dims {assembly.dims}'
    if assembly.dims.index(neuroid_dim) == 0:
        assembly = assembly.transpose('presentation', 'neuroid')

    patterns = assembly.values
    n_conds, n_chan = patterns.shape
    assert n_conds == 24, f'Expected 24 conditions, got {n_conds}'

    # remove units with no variance across conditions
    patterns_std = patterns.std(axis=0)
    patterns = patterns[:, patterns_std != 0]

    # select units with the highest mean response magnitude across conditions
    patterns_mean = patterns.mean(axis=0)
    std_x_units = patterns_mean.std()  # std of mean unit-wise response
    selected_units = np.abs(patterns_mean) > (std_x_units * 3.1)
    assert selected_units.any(), \
        'No units with mean response > 3.1 * std over all units'
    patterns = patterns[:, selected_units]

    # convert to z-score
    patterns_mean = np.tile(patterns.mean(0), (n_conds, 1))
    patterns_std = np.tile(patterns.std(0), (n_conds, 1))
    patterns = (patterns - patterns_mean) / patterns_std

    # perform pairwise correlation
    correlations = np.corrcoef(patterns)
    coords = {coord: coord_value for coord, coord_value in
              assembly.coords.items() if coord != neuroid_dim}
    dims = [dim if dim != neuroid_dim else assembly.dims[
        (i - 1) % len(assembly.dims)]
            for i, dim in enumerate(assembly.dims)]
    similarities = DataAssembly(correlations, coords=coords, dims=dims)
    return similarities


def _Coggan2024_Region(region: str):
    assembly = load_dataset('Coggan2024_fMRI')
    assembly = assembly.sel(region=region)
    assembly['region'] = ('subject', [region] * len(assembly['subject']))
    benchmark = Coggan2024_fMRI_Benchmark(
        identifier=f'tong.Coggan2024_fMRI.{region}-rdm',
        version=1,
        assembly=assembly,
        visual_degrees=9,
        ceiling_func=get_ceiling,
        parent=region,
        bibtex=BIBTEX)
    return benchmark


def Coggan2024_V1():
    return _Coggan2024_Region(region='V1')


def Coggan2024_V2():
    return _Coggan2024_Region(region='V2')


def Coggan2024_V4():
    return _Coggan2024_Region(region='V4')


def Coggan2024_IT():
    return _Coggan2024_Region(region='IT')

