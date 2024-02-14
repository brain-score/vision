import numpy as np
import pytest

from brainio.stimuli import StimulusSet
from brainscore_vision import load_stimulus_set, load_dataset 
from brainscore_vision.model_interface import BehavioralAssembly



@pytest.mark.memory_intense
@pytest.mark.private_access
class TestHebart2023:
    assembly = load_dataset('Hebart2023')
    stimulus_set = load_stimulus_set("Hebart2023")

    def test_assembly(self):
        stimulus_id = self.assembly.coords["stimulus_id"]
        triplet_id = self.assembly.coords["triplet_id"]
        assert len(stimulus_id) == len(triplet_id) == 453642
        assert len(np.unique(stimulus_id)) == 1854

        image_1 = self.assembly.coords["image_1"]
        image_2 = self.assembly.coords["image_2"]
        image_3 = self.assembly.coords["image_3"]
        assert len(image_1) == len(image_2) == len(image_3) == 453642

    def test_assembly_stimulusset_ids_match(self):
        stimulusset_ids = self.stimulus_set['stimulus_id']
        for assembly_stimulusid in ['image_1', 'image_2', 'image_3']:
            assembly_values = self.assembly[assembly_stimulusid].values
            assert set(assembly_values) == set(stimulusset_ids), \
                f"Assembly stimulus id reference '{assembly_stimulusid}' does not match stimulus_set"

    def test_stimulus_set(self):
        assert len(self.stimulus_set) == 1854
        assert {'unique_id', 'stimulus_id', 'filename',
                'WordNet_ID', 'Wordnet_ID2', 'Wordnet_ID3', 'Wordnet_ID4', 'WordNet_synonyms',
                'freq_1', 'freq_2', 'top_down_1', 'top_down_2', 'bottom_up', 'word_freq', 'word_freq_online',
                'example_image', 'dispersion', 'dominant_part', 'rank'} == set(self.stimulus_set.columns)
        assert isinstance(self.stimulus_set, StimulusSet)
