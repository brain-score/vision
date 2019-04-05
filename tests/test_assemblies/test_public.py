from brainscore.assemblies.public import load_assembly


def test_Majaj2015ITLowvar():
    assembly = load_assembly('dicarlo.Majaj2015.lowvar.IT')
    assert set(assembly['region'].values) == {'IT'}
    assert len(assembly['presentation']) == 3200
    assert len(assembly['neuroid']) == 168


def test_majaj2015TemporalLowvar():
    assembly = load_assembly('dicarlo.Majaj2015.temporal.lowvar')
    assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var03'
    assert len(assembly['presentation']) == 3200
    assert len(assembly['neuroid']) == 256
    assert len(assembly.sel(region='IT')['neuroid']) == 168
    assert len(assembly.sel(region='V4')['neuroid']) == 88


def test_majaj2015ITTemporalLowvar():
    assembly = load_assembly('dicarlo.Majaj2015.temporal.lowvar.IT')
    assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var03'
    assert set(assembly['region'].values) == {'IT'}
    assert len(assembly['presentation']) == 3200
    assert len(assembly['neuroid']) == 168


def test_majaj2015V4TemporalLowvar():
    assembly = load_assembly('dicarlo.Majaj2015.temporal.lowvar.V4')
    assert assembly.attrs['stimulus_set_name'] == 'dicarlo.hvm-var03'
    assert set(assembly['region'].values) == {'V4'}
    assert len(assembly['presentation']) == 3200
    assert len(assembly['neuroid']) == 88
