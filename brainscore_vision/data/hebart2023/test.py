import brainscore
import numpy as np
import pytest
from brainio.stimuli import StimulusSet


@pytest.mark.memory_intense
@pytest.mark.private_access
class TestHebart2023:
    def test_assembly(self):
        assembly = brainscore.get_assembly('Hebart2023')

        stimulus_id = assembly.coords["stimulus_id"]
        triplet_id = assembly.coords["triplet_id"]
        assert len(stimulus_id) == len(triplet_id) == 453642
        assert len(np.unique(stimulus_id)) == 1854

        image_1 = assembly.coords["image_1"]
        image_2 = assembly.coords["image_2"]
        image_3 = assembly.coords["image_3"]
        assert len(image_1) == len(image_2) == len(image_3) == 453642

    def test_assembly_stimulusset_ids_match(self):
        stimulus_set = brainscore.get_stimulus_set("Hebart2023")
        assembly = brainscore.get_assembly('Hebart2023')

        stimulusset_ids = stimulus_set['stimulus_id']
        for assembly_stimulusid in ['image_1', 'image_2', 'image_3']:
            assembly_values = assembly[assembly_stimulusid].values
            assert set(assembly_values) == set(stimulusset_ids), \
                f"Assembly stimulus id reference '{assembly_stimulusid}' does not match stimulus_set"

    def test_stimulus_set(self):
        stimulus_set = brainscore.get_stimulus_set("Hebart2023")
        assert len(stimulus_set) == 1854
        assert {'unique_id', 'stimulus_id', 'filename',
                'WordNet_ID', 'Wordnet_ID2', 'Wordnet_ID3', 'Wordnet_ID4', 'WordNet_synonyms',
                'freq_1', 'freq_2', 'top_down_1', 'top_down_2', 'bottom_up', 'word_freq', 'word_freq_online',
                'example_image', 'dispersion', 'dominant_part', 'rank'} == set(stimulus_set.columns)
        assert isinstance(stimulus_set, StimulusSet)
