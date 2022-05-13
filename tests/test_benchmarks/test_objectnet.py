import numpy as np
import pytest

from brainio.assemblies import BehavioralAssembly
from brainscore.benchmarks.objectnet import Objectnet
from brainscore.model_interface import BrainModel

@pytest.mark.private_access
class TestObjectnet:
    def test_groundtruth(self):
        benchmark = Objectnet()
        source = benchmark._stimulus_set

        class GroundTruth(BrainModel):
            def start_task(self, task, fitting_stimuli):
                assert task == BrainModel.Task.label
                assert fitting_stimuli == 'imagenet'  # shortcut

            def look_at(self, stimuli, number_of_trials=1):
                source_image_ids = source['image_id'].values
                stimuli_image_ids = stimuli['image_id'].values
                sorted_x = source_image_ids[np.argsort(source_image_ids)]
                sorted_index = np.searchsorted(sorted_x, stimuli_image_ids)
                aligned_source = source.loc[sorted_index]
                labels = aligned_source['synset'].values
                return BehavioralAssembly([labels], coords={
                    **{column: ('presentation', aligned_source[column].values) for column in aligned_source.columns},
                    **{'choice': ('choice', ['dummy'])}}, dims=['choice', 'presentation'])

        candidate = GroundTruth()
        score = benchmark(candidate)
        assert score.sel(aggregation='center') == pytest.approx(1)
