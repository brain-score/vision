import os

import brainscore
import pytest
import numpy as np

import brainio


class TestStimulusSet:
    def __init__(self, name, stimuli_type, num_stimuli):
        self.name = name
        self.stimuli_type = stimuli_type
        self.num_stimuli = num_stimuli

    def __call__(self, stimulus_set):
        self.test_list_stimulus_set()
        self.test_stimuli_type()
        self.test_num_stimuli()

    # ensure the stimulus set exists:
    def test_list_stimulus_set(self):
        l = brainio.list_stimulus_sets()
        assert self.name in l, f"Stimulus Set with name {self.name}not found in BrainIO."

    # ensure that the type of stimuli (image, video, etc) matches the input
    def test_stimuli_type(self):
        stimulus_set = brainio.get_stimulus_set(self.name)
        assert set(stimulus_set["stimulus_type"]) == {self.stimuli_type}, \
            f"Parameter stimuli_type={self.stimuli_type} does not match packaged stimuli_type."

    # ensure the number of stimuli is equal to the input parameter:
    def test_num_stimuli(self):
        stimulus_set = brainio.get_stimulus_set(self.name)
        assert len(stimulus_set) == self.num_stimuli, \
            f"parameter num_stimuli={self.num_stimuli} does not match length of stimulus set (length {len(stimulus_set)})."


class TestAssembly:
    def __init__(self, assembly_name, num_subjects, num_reps, num_stimuli):
        self.assembly_name = assembly_name
        self.num_subjects = num_subjects
        self.num_reps = num_reps
        self.num_stimuli = num_stimuli
        self.projected_size = num_subjects * num_reps * num_stimuli

    def __call__(self, stimulus_set):
        self.test_list_assembly()
        self.test_size()
        self.test_num_subjects()
        self.test_num_stimuli()

    # test for assembly existence
    def test_list_assembly(self):
        l = brainio.list_assemblies()
        assert self.assembly_name in l, f"Assembly with name {self.assembly_name} not found in BrainIO."

    # make sure assembly dim is correct length
    def test_size(self):
        assembly = brainio.get_assembly(self.assembly_name)
        assert len(assembly['presentation']) == self.projected_size, \
            f"Calculated assembly size ({self.num_subjects} * {self.num_stimuli} * {self.num_reps} = " \
            f"{self.projected_size}) does not match actual assembly size ({len(assembly)})."

    # check number of subjects:
    def test_num_subjects(self):
        assembly = brainio.get_assembly(self.assembly_name)
        assert len(np.unique(assembly['subject_id'].values)) == self.num_subjects, f"Parameter num_subjects does not " \
                                                                                   f"match num_subjects in assembly. "

    # check number of stimuli:
    def test_num_stimuli(self):
        assembly = brainio.get_assembly(self.assembly_name)
        assert len(np.unique(assembly['stimulus_id'].values)) == self.num_stimuli, f"Parameter num_stimuli does not " \
                                                                                   f"match num_stimuli in assembly. "
