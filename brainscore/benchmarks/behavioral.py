import numpy as np

import brainscore
from brainscore.metrics import Score

from brainscore.benchmarks import BenchmarkBase
from brainscore.assemblies.private import load_assembly
from brainscore.metrics.behavior import I2n
from brainscore.metrics.transformations import apply_aggregate
from brainscore.model_interface import BrainModel


class BehavioralBenchmark(BenchmarkBase):
    def __init__(self, identifier, fitting_stimuli, assembly, metric, ceiling_skipna=True):
        self._metric = metric
        self._fitting_stimuli = fitting_stimuli
        self._assembly = assembly
        super(BehavioralBenchmark, self).__init__(
            identifier=identifier,
            ceiling_func=lambda: self._metric.ceiling(self._assembly, skipna=ceiling_skipna))

    def __call__(self, candidate: BrainModel):
        candidate.start_task(BrainModel.Task.probabilities, self._fitting_stimuli)
        probabilities = candidate.look_at(self._assembly.stimulus_set)
        score = self._metric(probabilities, self._assembly)
        score = self.ceil_score(score, self.ceiling)
        return score

    def ceil_score(self, score, ceiling):
        assert set(score.raw['split'].values) == set(ceiling.raw['split'].values)
        split_scores = []
        for split in ceiling.raw['split'].values:
            split_score = score.raw.sel(split=split)
            split_ceiling = ceiling.raw.sel(split=split)
            ceiled_split_score = split_score / np.sqrt(split_ceiling)
            ceiled_split_score = ceiled_split_score.expand_dims('split')
            ceiled_split_score['split'] = [split]
            split_scores.append(ceiled_split_score)
        split_scores = Score.merge(*split_scores)
        split_scores = apply_aggregate(self._metric.aggregate, split_scores)
        split_scores.attrs[Score.RAW_VALUES_KEY] = score  # this will override raw per-split ceiled scores which is ok
        split_scores.attrs['ceiling'] = ceiling
        return split_scores


def DicarloRajalingham2018I2n():
    fitting_stimuli = brainscore.get_stimulus_set('dicarlo.objectome.public')
    assembly = load_assembly('dicarlo.Rajalingham2018')
    return BehavioralBenchmark(identifier='dicarlo.Rajalingham2018-i2n',
                               fitting_stimuli=fitting_stimuli, assembly=assembly, metric=I2n())


def DicarloKarCocoI2n():
    # fitting_stimuli = brainscore.get_stimulus_set('dicarlo.karcoco.public')
    # assembly = load_assembly('dicarlo.KarCoco')

    import pandas as pd
    import xarray as xr
    from brainio_base.assemblies import BehavioralAssembly
    from brainio_base.stimuli import StimulusSet
    files_dir = '/braintree/home/msch/share/KarCoco/'
    fitting_stimuli = pd.read_pickle(f"{files_dir}/image_dicarlo_karcoco_public.pkl")
    testing_stimuli = pd.read_pickle(f"{files_dir}/image_dicarlo_karcoco_private.pkl")
    fitting_name, testing_name = fitting_stimuli.name, testing_stimuli.name
    fitting_stimuli, testing_stimuli = StimulusSet(fitting_stimuli), StimulusSet(testing_stimuli)
    fitting_stimuli.name, testing_stimuli.name = fitting_name, testing_name
    fitting_stimuli.image_paths = {row.image_id: row.image_current_local_file_path
                                   for row in fitting_stimuli.itertuples()}
    testing_stimuli.image_paths = {row.image_id: row.image_current_local_file_path
                                   for row in testing_stimuli.itertuples()}
    image_id = testing_stimuli['image_id'].values[0]
    testing_stimuli.get_image(image_id)
    assembly = xr.open_dataarray(f"{files_dir}/assy_dicarlo_karcoco_private.nc")
    assembly = BehavioralAssembly(assembly)
    assembly.attrs['stimulus_set'] = testing_stimuli
    assembly.attrs['stimulus_set_name '] = testing_stimuli.name
    assembly.stimulus_set.get_image(image_id)
    return BehavioralBenchmark(identifier='dicarlo.KarCoco-i2n',
                               fitting_stimuli=fitting_stimuli, assembly=assembly, metric=I2n(), ceiling_skipna=True)
