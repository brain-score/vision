import numpy as np
from numpy.random.mtrand import RandomState
from sklearn.model_selection import StratifiedShuffleSplit

from brainio_collection.fetch import fetch_assembly, get_assembly
from brainio_contrib.packaging import package_data_assembly, package_stimulus_set


def adapt_stimulus_set(assembly, name_suffix):
    stimulus_set_name = f"{assembly.stimulus_set.name}-{name_suffix}"
    assembly.attrs['stimulus_set'] = assembly.stimulus_set[
        assembly.stimulus_set['image_id'].isin(assembly['image_id'].values)]
    assembly.stimulus_set.name = stimulus_set_name
    assembly.attrs['stimulus_set_name'] = stimulus_set_name


def load_assembly(assembly_name):
    assembly = get_assembly(assembly_name)

    if not hasattr(assembly.stimulus_set, 'name'):
        assembly.stimulus_set.name = assembly.stimulus_set_name

    stimulus_set_degrees = {'dicarlo.hvm': 8, 'movshon.FreemanZiemba2013': 4}
    if assembly.stimulus_set.name in stimulus_set_degrees:
        assembly.stimulus_set['degrees'] = stimulus_set_degrees[assembly.stimulus_set.name]
    return assembly


def package_Movshon_datasets(name):
    assembly = load_assembly(name)
    assembly.load()
    base_assembly = assembly
    _, unique_indices = np.unique(base_assembly['image_id'].values, return_index=True)
    unique_indices = np.sort(unique_indices)  # preserve order
    image_ids = base_assembly['image_id'].values[unique_indices]
    stratification_values = base_assembly['texture_type'].values[unique_indices]
    rng = RandomState(seed=12)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=.3, test_size=None, random_state=rng)
    split = next(splitter.split(np.zeros(len(image_ids)), stratification_values))
    access_indices = {assembly_type: image_indices
                      for assembly_type, image_indices in zip(['public', 'private'], split)}
    for access in ['public', 'private']:
        indices = access_indices[access]
        subset_image_ids = image_ids[indices]
        assembly = base_assembly[
            {'presentation': [image_id in subset_image_ids for image_id in base_assembly['image_id'].values]}]
        adapt_stimulus_set(assembly, access)
        package_stimulus_set(assembly.attrs['stimulus_set'], stimulus_set_name=assembly.attrs['stimulus_set_name'])
        del assembly.attrs['stimulus_set']
        package_data_assembly(assembly, f"{name}.{access}", stimulus_set_name=assembly.attrs['stimulus_set_name'])

    # not really sure if this is necessary
    return assembly


if __name__ == '__main__':
    package_Movshon_datasets(name='movshon.FreemanZiemba2013')
