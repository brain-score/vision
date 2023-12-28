import numpy as np
import pytest

import brainscore
from brainio.stimuli import StimulusSet


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestHebart2023:
    assembly = brainscore.get_assembly('Hebart2023')

    def test_assembly(self):
        stimulus_id = self.assembly.coords["stimulus_id"]
        triplet_id = self.assembly.coords["triplet_id"]
        assert len(stimulus_id) == len(triplet_id) == 453642
        assert len(np.unique(stimulus_id)) == 1854

        image_1 = self.assembly.coords["image_1"]
        image_2 = self.assembly.coords["image_2"]
        image_3 = self.assembly.coords["image_3"]
        assert len(image_1) == len(image_2) == len(image_3) == 453642

    def test_stimulus_set(self):
        stimulus_set = self.assembly.attrs['stimulus_set']
        assert len(stimulus_set) == 1854
        assert set(stimulus_set.columns) == {
            'stimulus_id', 'image_no', 'filename',
            'WordNet_ID', 'Wordnet_ID2', 'Wordnet_ID3', 'Wordnet_ID4', 'WordNet_Synonyms',
            'freq_1', 'freq_2', 'top_down_1', 'top_down_2', 'bottom_up', 'word_freq', 'word_freq_online',
            'example_image', 'dispersion', 'dominant_part', 'rank'}
        assert isinstance(stimulus_set, StimulusSet)
