import numpy as np
import pytest
from brainscore_vision import load_dataset, load_stimulus_set
from brainscore_vision.benchmark_helpers import check_standard_format


# testing stimulus sets
@pytest.mark.private_access
class TestStimulusSets:
    # test stimulus_set data:
    @pytest.mark.parametrize('identifier', [
        'circle_line', 'color', 'convergence', 'eighth',
        'gray_easy', 'gray_hard', 'half', 'juncture',
        'lle', 'llh', 'quarter', 'round_f',
        'round_v', 'tilted_line'
    ])
    def test_stimulus_set_exist(self, identifier):
        stimulus_set = load_stimulus_set(f"Ferguson2024_{identifier}")
        assert stimulus_set is not None
        assert stimulus_set.identifier == f"Ferguson2024_{identifier}"

    # test the number of images
    @pytest.mark.parametrize('identifier', [
        'circle_line', 'color', 'convergence', 'eighth',
        'gray_easy', 'gray_hard', 'half', 'juncture',
        'lle', 'llh', 'quarter', 'round_f',
        'round_v', 'tilted_line'
    ])
    def test_num_images(self, identifier):
        stimulus_set = load_stimulus_set(f"Ferguson2024_{identifier}")
        assert len(np.unique(stimulus_set['stimulus_id'].values)) == 48

    # test the number of blocks:
    @pytest.mark.parametrize('identifier', [
        'circle_line', 'color', 'convergence', 'eighth',
        'gray_easy', 'gray_hard', 'half', 'juncture',
        'lle', 'llh', 'quarter', 'round_f',
        'round_v', 'tilted_line'
    ])
    def test_num_blocks(self, identifier):
        stimulus_set = load_stimulus_set(f"Ferguson2024_{identifier}")
        assert len(np.unique(stimulus_set['block'].values)) == 2


# testing assemblies
@pytest.mark.private_access
class TestAssemblies:
    # test stimulus_set data:
    @pytest.mark.parametrize('identifier', [
        'circle_line', 'color', 'convergence', 'eighth',
        'gray_easy', 'gray_hard', 'half', 'juncture',
        'lle', 'llh', 'quarter', 'round_f',
        'round_v', 'tilted_line'
    ])
    def test_exist_and_alignment(self, identifier):
        assembly = load_dataset(f"Ferguson2024_{identifier}")
        assert assembly is not None
        assert assembly.identifier == f"Ferguson2024_{identifier}"
        assert assembly.stimulus_set.identifier == f"Ferguson2024_{identifier}"

    # test the number of images
    @pytest.mark.parametrize('identifier', [
        'circle_line', 'color', 'convergence', 'eighth',
        'gray_easy', 'gray_hard', 'half', 'juncture',
        'lle', 'llh', 'quarter', 'round_f',
        'round_v', 'tilted_line'
    ])
    def test_distinct_values(self, identifier):
        assembly = load_dataset(f"Ferguson2024_{identifier}")
        assert set(assembly['block'].values) == {"first", "second"}
        assert set(assembly['keypress_response'].values) == {"f", "j"}
        assert set(assembly['trial_type'].values) == {"normal"}
        assert set(assembly['distractor_nums'].values) == {"1.0", "5.0", "11.0"}
        assert set(assembly['target_present'].values) == {True, False}
        assert set(assembly['correct'].values) == {True, False}

    # test the number of subjects
    @pytest.mark.parametrize('identifier, num_subjects', [
        ('circle_line', 30),
        ('color', 29),
        ('convergence', 27),
        ('eighth', 30),
        ('gray_easy', 28),
        ('gray_hard', 29),
        ('half', 29),
        ('juncture', 27),
        ('lle', 29),
        ('llh', 28),
        ('quarter', 28),
        ('round_f', 30),
        ('round_v', 29),
        ('tilted_line', 30),
    ])
    def test_num_subjects(self, identifier, num_subjects):
        assembly = load_dataset(f"Ferguson2024_{identifier}")
        assert set(assembly['participant_id'].values) == num_subjects

    # test the number of rows (size)
    @pytest.mark.parametrize('identifier, size', [
        ('circle_line', 4292),
        ('color', 4132),
        ('convergence', 3874),
        ('eighth', 4302),
        ('gray_easy', 4047),
        ('gray_hard', 4143),
        ('half', 4162),
        ('juncture', 3876),
        ('lle', 4167),
        ('llh', 4166),
        ('quarter', 4050),
        ('round_f', 4380),
        ('round_v', 4257),
        ('tilted_line', 4314),
    ])
    def test_num_subjects(self, identifier, size):
        assembly = load_dataset(f"Ferguson2024_{identifier}")
        assert len(assembly) == size


# testing training sets
@pytest.mark.private_access
class TestTrainingStimulusSets:
    # test stimulus_set data:
    @pytest.mark.parametrize('identifier', [
        'circle_line', 'color', 'convergence', 'eighth',
        'gray_easy', 'gray_hard', 'half', 'juncture',
        'lle', 'llh', 'quarter', 'round_f',
        'round_v', 'tilted_line'
    ])
    def test_stimulus_set_exist(self, identifier):
        stimulus_set = load_stimulus_set(f"Ferguson2024_{identifier}_training_stimuli")
        assert stimulus_set is not None
        assert stimulus_set.identifier == f"Ferguson2024_{identifier}_training_stimuli"

    # test the number of images
    @pytest.mark.parametrize('identifier', [
        'circle_line', 'color', 'convergence', 'eighth',
        'gray_easy', 'gray_hard', 'half', 'juncture',
        'lle', 'llh', 'quarter', 'round_f',
        'round_v', 'tilted_line'
    ])
    def test_num_images(self, identifier):
        stimulus_set = load_stimulus_set(f"Ferguson2024_{identifier}_training_stimuli")
        assert len(np.unique(stimulus_set['stimulus_id'].values)) == 1920

    # test the number of blocks:
    @pytest.mark.parametrize('identifier', [
        'circle_line', 'color', 'convergence', 'eighth',
        'gray_easy', 'gray_hard', 'half', 'juncture',
        'lle', 'llh', 'quarter', 'round_f',
        'round_v', 'tilted_line'
    ])
    def test_num_blocks(self, identifier):
        stimulus_set = load_stimulus_set(f"Ferguson2024_{identifier}_training_stimuli")
        assert len(np.unique(stimulus_set['block'].values)) == 2
