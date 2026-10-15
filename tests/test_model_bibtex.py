from brainscore_vision import load_model_bibtex


def test_load_model_bibtex():
    assert 'doi={10.1038/s41467-020-19632-w}' in load_model_bibtex('mehrer2020_alexnet_training_seed_01')


def test_load_model_bibtex_unknown_model():
    assert load_model_bibtex('no_such_model_identifier') is None
