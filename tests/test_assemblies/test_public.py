from brainscore.assemblies.public import load_assembly


def test_Majaj2015ITLowvar():
    assembly = load_assembly('dicarlo.Majaj2015.lowvar.IT')
    assert set(assembly['region'].values) == {'IT'}
    assert len(assembly['presentation']) == 3200
    assert len(assembly['neuroid']) == 168
