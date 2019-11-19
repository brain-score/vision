from brainscore.public_benchmarks import list_public_assemblies


def test_list():
    assemblies = list_public_assemblies()
    assert set(assemblies) == {'dicarlo.Majaj2015.public', 'dicarlo.Majaj2015.temporal.public',
                               'movshon.FreemanZiemba2013.public', 'dicarlo.Rajalingham2018.public'}
