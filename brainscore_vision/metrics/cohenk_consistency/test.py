import numpy as np
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import load_metric


def test_score():
    assembly_1, assembly_2 = _make_data()
    metric = load_metric('cohenk_consistency')
    score = metric(assembly_1, assembly_2)
    assert score == approx(0.5)


def test_has_error():
    assembly_1, assembly_2 = _make_data()
    metric = load_metric('cohenk_consistency')
    score = metric(assembly_1, assembly_2)    
    assert hasattr(score, 'error')


def _make_data():
    # subject A is 5 / 9 = 0.55...% correct
    # subject B is 4 / 9 = 0.44...% correct
    # subject C is 9 / 9 = 100% correct

    assembly_2 = BehavioralAssembly([1,0,1,1,0,0,1,0],
                               coords={
                                   'stimulus_id': ('presentation', ['stim_1', 'stim_2', 'stim_3', 'stim_4',
                                                                   'stim_5', 'stim_6', 'stim_7', 'stim_8',
                                                                   ]),
                                   'scenario': ('presentation', ['s1', 's1', 's2', 's2', 's3', 's3','s4', 's4']),
                                   'gameID': ('presentation', ['0', '0', '1', '1', '2', '2', '3', '3',]),
                                   'responseBool': ('presentation', [1, 0, 1, 1, 0, 1, 1, 0]),
                               },
                                    dims=['presentation'])
    
    assembly_1 = BehavioralAssembly([0.7, 0.2, 0.7, 0.2, 0, 0, 1, 0],
                                       coords=
                                       {'stimulus_id': ('presentation', ['stim_1_img.mp4', 'stim_2_img.mp4', 'stim_3_img.mp4', 'stim_4_img.mp4',
                                                                   'stim_5_img.mp4', 'stim_6_img.mp4', 'stim_7_img.mp4', 'stim_8_img.mp4']),
                                        'choice': ('presentation', [1, 0, 1, 0, 0, 0, 1, 0]), 
                                        'scenario': ('presentation', ['s1', 's1', 's2', 's2', 's3', 's3','s4', 's4'])},
                                       dims=['presentation'])

    return assembly_1, assembly_2


test_score()