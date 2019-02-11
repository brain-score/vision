import brainscore


def _majaj2015(region):
    assembly = brainscore.get_assembly('dicarlo.Majaj2015').sel(region=region, variation=6).squeeze('time_bin')
    stimulus_set_name = f"{assembly.stimulus_set.name}-V6"
    assembly.attrs['stimulus_set'] = assembly.stimulus_set[
        assembly.stimulus_set['image_id'].isin(assembly['image_id'].values)]
    assembly.stimulus_set.name = stimulus_set_name
    assembly.attrs['stimulus_set_name'] = stimulus_set_name
    assembly.name = f"{assembly.name}-V6-{region}"
    return assembly


def v4_translation_data():
    return _majaj2015('V4')


def it_translation_data():
    return _majaj2015('IT')
