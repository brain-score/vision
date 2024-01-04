from pytest import approx

from brainscore_vision import load_metric, load_dataset


def test_identity_circularvariance():
    assembly = load_dataset('Ringach2002')
    metric = load_metric('ks_similarity', property_name='circular_variance')
    score = metric(assembly, assembly)
    assert score == approx(1, abs=0.05)
