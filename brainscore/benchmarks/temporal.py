import numpy as np
from brainscore.benchmarks import BenchmarkBase

from brainscore.assemblies.private import assembly_loaders
from brainscore.benchmarks.regressing import build_benchmark
from brainscore.metrics import Score
from brainscore.metrics.ceiling import InternalConsistency, TemporalCeiling
from brainscore.metrics.ost import OSTCorrelation
from brainscore.metrics.regression import CrossRegressedCorrelation, pearsonr_correlation, pls_regression
from brainscore.metrics.temporal import TemporalRegressionAcrossTime, TemporalCorrelationAcrossImages
from brainscore.model_interface import BrainModel


class DicarloKar2019OST(BenchmarkBase):
    def __init__(self):
        super(DicarloKar2019OST, self).__init__(identifier='dicarlo.Kar2019-ost',
                                                ceiling_func=lambda x: Score(0.79))  # ceiling computed independently

        import pandas as pd
        import xarray as xr
        from brainio_base.assemblies import DataAssembly
        from brainio_base.stimuli import StimulusSet
        files_dir = '/braintree/home/msch/share/Kar2019OST/'
        stimuli = pd.read_pickle(f"{files_dir}/image_dicarlo_kar2019.pkl")
        stimuli_name = stimuli.name
        stimuli = StimulusSet(stimuli)
        stimuli.name = stimuli_name
        stimuli.image_paths = {row.image_id: row.image_current_local_file_path for row in stimuli.itertuples()}
        assembly = xr.open_dataarray(f"{files_dir}/assy_dicarlo_kar2019.nc")
        assembly = DataAssembly(assembly)
        assembly.attrs['stimulus_set'] = stimuli
        assembly.attrs['stimulus_set_name '] = stimuli.name
        # drop duplicate images
        _, index = np.unique(assembly['image_id'], return_index=True)
        assembly = assembly.isel(presentation=index)
        assembly.stimulus_set = assembly.stimulus_set.drop_duplicates('image_id')

        assembly = assembly.sel(decoder='logistic')

        self._assembly = assembly
        self._assembly['truth'] = self._assembly['image_label']
        self._assembly.stimulus_set['truth'] = self._assembly.stimulus_set['image_label']

        self._similarity_metric = OSTCorrelation()

    def __call__(self, candidate: BrainModel):
        time_bins = [(time_bin_start, time_bin_start + 10) for time_bin_start in range(70, 250, 10)]
        candidate.start_recording('IT', time_bins=time_bins)
        recordings = candidate.look_at(self._assembly.stimulus_set)
        score = self._similarity_metric(recordings, self._assembly)
        return score


class TimeFilteredAssemblyLoader:
    def __init__(self, baseloader, time_bins):
        self._loader = baseloader
        self._time_bins = time_bins

    def __call__(self, *args, **kwargs):
        assembly = self._loader(*args, **kwargs)
        assembly = assembly.sel(time_bin=self._time_bins)
        return assembly


def _DicarloMajaj2015TemporalRegion(region):
    metric = CrossRegressedCorrelation(regression=TemporalRegressionAcrossTime(regression=pls_regression()),
                                       correlation=TemporalCorrelationAcrossImages(correlation=pearsonr_correlation()))
    # sub-select time-bins, and get rid of overlapping time bins
    time_bins = [(time_bin_start, time_bin_start + 20) for time_bin_start in range(0, 231, 20)]
    loader = TimeFilteredAssemblyLoader(assembly_loaders[f'dicarlo.Majaj2015.temporal.highvar.{region}'], time_bins)
    return build_benchmark(identifier=f'dicarlo.Majaj2015.temporal.{region}', assembly_loader=loader,
                           similarity_metric=metric, ceiler=TemporalCeiling(InternalConsistency()))


DicarloMajaj2015TemporalV4PLS = lambda: _DicarloMajaj2015TemporalRegion(region='V4')
DicarloMajaj2015TemporalITPLS = lambda: _DicarloMajaj2015TemporalRegion(region='IT')


def _MovshonFreemanZiemba2013TemporalRegion(region):
    metric = CrossRegressedCorrelation(regression=TemporalRegressionAcrossTime(regression=pls_regression()),
                                       correlation=TemporalCorrelationAcrossImages(correlation=pearsonr_correlation()),
                                       crossvalidation_kwargs=dict(stratification_coord='texture_type'))
    # sub-select time-bins, and get rid of overlapping time bins
    time_bins = [(time_bin_start, time_bin_start + 10) for time_bin_start in range(0, 241, 10)]
    loader = assembly_loaders[f'movshon.FreemanZiemba2013.temporal.private.{region}']
    loader = TimeFilteredAssemblyLoader(loader, time_bins)
    return build_benchmark(identifier=f'movshon.FreemanZiemba2013.temporal.{region}', assembly_loader=loader,
                           similarity_metric=metric, ceiler=TemporalCeiling(InternalConsistency()))


MovshonFreemanZiemba2013TemporalV1PLS = lambda: _MovshonFreemanZiemba2013TemporalRegion(region='V1')
MovshonFreemanZiemba2013TemporalV2PLS = lambda: _MovshonFreemanZiemba2013TemporalRegion(region='V2')
