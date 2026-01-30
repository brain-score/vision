from brainio.assemblies import NeuroidAssembly, walk_coords
from brainscore_core import Score

import brainscore_vision
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.metrics.spatial_correlation.metric import inv_ks_similarity
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import LazyLoad
from ..majajhong2015.benchmark import BIBTEX

SPATIAL_BIN_SIZE_MM = .1  # .1 mm is an arbitrary choice


class DicarloMajajHong2015ITSpatialCorrelation(BenchmarkBase):
    def __init__(self):
        """
        This benchmark compares the distribution of pairwise response correlation as a function of distance between the
            data recorded in Majaj* and Hong* et al. 2015 and a candidate model.
        """
        self._assembly = self._load_assembly()
        self._metric = brainscore_vision.load_metric(
            'spatial_correlation',
            similarity_function=inv_ks_similarity,
            bin_size_mm=SPATIAL_BIN_SIZE_MM,
            num_bootstrap_samples=100_000,
            num_sample_arrays=10)
        ceiler = brainscore_vision.load_metric('inter_individual_helper', self._metric.compare_statistics)
        target_statistic = LazyLoad(lambda: self._metric.compute_global_tissue_statistic_target(self._assembly))
        super().__init__(identifier='dicarlo.MajajHong2015.IT-spatial_correlation',
                         ceiling_func=lambda: ceiler(target_statistic),
                         version=1,
                         parent='IT',
                         bibtex=BIBTEX)

    def _load_assembly(self) -> NeuroidAssembly:
        assembly = brainscore_vision.load_dataset('MajajHong2015').sel(region='IT')
        assembly = self.squeeze_time(assembly)
        assembly = self.tissue_update(assembly)
        return assembly

    def __call__(self, candidate: BrainModel) -> Score:
        """
        This computes the statistics, i.e. the pairwise response correlation of candidate and target, respectively and
        computes a ceiling-normalized score based on the ks similarity of the two resulting distributions.
        :param candidate: BrainModel
        :return: average inverted ks similarity for the pairwise response correlation compared to the MajajHong assembly
        """
        candidate.start_recording(recording_target=BrainModel.RecordingTarget.IT,
                                  time_bins=[(70, 170)],
                                  # "we implanted each monkey with three arrays in the left cerebral hemisphere"
                                  )
        candidate_assembly = candidate.look_at(self._assembly.stimulus_set)
        candidate_assembly = self.squeeze_time(candidate_assembly)

        raw_score = self._metric(candidate_assembly, self._assembly)
        score = raw_score / self.ceiling
        score.attrs['raw'] = raw_score
        score.attrs['ceiling'] = self.ceiling
        return score

    @staticmethod
    def tissue_update(assembly):
        """
        The current MajajHong2015 assembly has x and y coordinates of each array electrode stored as
        coordinates `x` and `y` rather than the preferred `tissue_x` and `tissue_y`. Add these coordinates here.
        """
        if not hasattr(assembly, 'tissue_x'):
            assembly['tissue_x'] = assembly['x']
            assembly['tissue_y'] = assembly['y']
        # re-index
        attrs = assembly.attrs
        assembly = type(assembly)(assembly.values, coords={
            coord: (dims, values) for coord, dims, values in walk_coords(assembly)}, dims=assembly.dims)
        assembly.attrs = attrs
        return assembly

    @staticmethod
    def squeeze_time(assembly):
        if 'time_bin' in assembly.dims:
            assembly = assembly.squeeze('time_bin')
        if hasattr(assembly, "time_step"):
            assembly = assembly.squeeze("time_step")
        return assembly
