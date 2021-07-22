import numpy as np
from brainio_base.assemblies import walk_coords, array_is_element
from numpy.random.mtrand import RandomState
from sklearn.model_selection import StratifiedShuffleSplit
from xarray import DataArray

from brainio_collection.fetch import fetch_assembly, get_assembly
from brainio_collection.transform import subset
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


def apply_keep_attrs(assembly, fnc):  # workaround to keeping attrs
    attrs = assembly.attrs
    assembly = fnc(assembly)
    assembly.attrs = attrs
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


def _filter_erroneous_neuroids(assembly):
    err_neuroids = ['Tito_L_P_8_5', 'Tito_L_P_7_3', 'Tito_L_P_7_5', 'Tito_L_P_5_1', 'Tito_L_P_9_3',
                    'Tito_L_P_6_3', 'Tito_L_P_7_4', 'Tito_L_P_5_0', 'Tito_L_P_5_4', 'Tito_L_P_9_6',
                    'Tito_L_P_0_4', 'Tito_L_P_4_6', 'Tito_L_P_5_6', 'Tito_L_P_7_6', 'Tito_L_P_9_8',
                    'Tito_L_P_4_1', 'Tito_L_P_0_5', 'Tito_L_P_9_9', 'Tito_L_P_3_0', 'Tito_L_P_0_3',
                    'Tito_L_P_6_6', 'Tito_L_P_5_7', 'Tito_L_P_1_1', 'Tito_L_P_3_8', 'Tito_L_P_1_6',
                    'Tito_L_P_3_5', 'Tito_L_P_6_8', 'Tito_L_P_2_8', 'Tito_L_P_9_7', 'Tito_L_P_6_7',
                    'Tito_L_P_1_0', 'Tito_L_P_4_5', 'Tito_L_P_4_9', 'Tito_L_P_7_8', 'Tito_L_P_4_7',
                    'Tito_L_P_4_0', 'Tito_L_P_3_9', 'Tito_L_P_7_7', 'Tito_L_P_4_3', 'Tito_L_P_9_5']
    good_neuroids = [i for i, neuroid_id in enumerate(assembly['neuroid_id'].values)
                     if neuroid_id not in err_neuroids]
    assembly = assembly.isel(neuroid=good_neuroids)
    return assembly


def package_dicarlo_datasets(name):
    base_assembly = load_assembly(name)
    base_assembly.load()
    base_assembly = _filter_erroneous_neuroids(base_assembly)
    for variation_name, target_variation in {'public': [0, 3], 'private': [6]}.items():
        assembly = base_assembly[
            {'presentation': [variation in target_variation for variation in base_assembly['variation'].values]}]
        assert hasattr(assembly, 'variation')
        adapt_stimulus_set(assembly, name_suffix=variation_name)
        package_stimulus_set(assembly.attrs['stimulus_set'], stimulus_set_name=assembly.attrs['stimulus_set_name'],
                             bucket_name="brainio-dicarlo")
        del assembly.attrs['stimulus_set']
        package_data_assembly(assembly, f'{name}.{variation_name}',assembly.attrs['stimulus_set_name'], bucket_name='brainio-dicarlo')
    return assembly


if __name__ == '__main__':
    package_dicarlo_datasets(name='dicarlo.Majaj2015')
    package_dicarlo_datasets(name='dicarlo.Majaj2015.temporal')
    package_Movshon_datasets(name='movshon.FreemanZiemba2013')
