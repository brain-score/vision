from brainscore_vision import load_stimulus_set


def test_stimuli():
    stimulus_set = load_stimulus_set('Islam2021')
    assert len(set(stimulus_set["texture"])) == 5
    assert len(set(stimulus_set["shape"])) == 20
    assert len(stimulus_set) == 4369 * 5
