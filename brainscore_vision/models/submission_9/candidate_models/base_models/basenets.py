import glob
import os
import pickle

import numpy as np
import pandas as pd

import brainscore
from brainio_base.assemblies import NeuroidAssembly, walk_coords, array_is_element


def package(features_path='/braintree/data2/active/users/qbilius/computed/hvm/ait'):
    assert os.path.isdir(features_path)
    features_paths = [os.path.join(features_path, 'basenets_hvm_feats_V4'),
                      os.path.join(features_path, 'basenets_hvm_feats_pIT'),
                      os.path.join(features_path, 'basenets_hvm_feats')]

    # alignment
    meta = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'basenets-meta.pkl'))
    meta = meta[meta['var'] == 6]
    meta_ids = meta['id'].values.tolist()

    hvm = brainscore.get_assembly('dicarlo.Majaj2015') \
        .sel(variation=6) \
        .multi_groupby(['category_name', 'object_name', 'image_id']) \
        .mean(dim="presentation") \
        .squeeze("time_bin")
    hvm_ids = hvm['image_id'].values.tolist()

    assert len(hvm_ids) == len(meta_ids)
    indexes = [meta_ids.index(id) for id in hvm_ids]

    basenets = []
    for activations_path_v4 in glob.glob(os.path.join(features_paths[0], '*.npy')):
        activations_path_pit = os.path.abspath(os.path.join(features_paths[1], os.path.basename(activations_path_v4)))
        activations_path_ait = os.path.abspath(os.path.join(features_paths[2], os.path.basename(activations_path_v4)))
        assert os.path.isfile(activations_path_pit)
        assert os.path.isfile(activations_path_ait)
        print(activations_path_v4, activations_path_pit, activations_path_ait, end='')
        activations_v4 = np.load(activations_path_v4)
        activations_pit = np.load(activations_path_pit)
        activations_ait = np.load(activations_path_ait)
        assert activations_v4.shape[0] == activations_pit.shape[0] == activations_ait.shape[0] == len(indexes)
        activations_v4 = activations_v4[indexes, :]
        activations_pit = activations_ait[indexes, :]
        activations_ait = activations_ait[indexes, :]
        coords = {coord: (dims, values) for coord, dims, values in walk_coords(hvm)
                  if array_is_element(dims, 'presentation')}
        coords['neuroid_id'] = 'neuroid', list(range(3000))
        coords['layer'] = 'neuroid', np.concatenate([np.repeat('basenet-layer_v4', 1000),
                                                     np.repeat('basenet-layer_pit', 1000),
                                                     np.repeat('basenet-layer_ait', 1000)])
        activations = np.concatenate([activations_v4, activations_pit, activations_ait], axis=1)
        print(activations.shape, end='')
        assert activations.shape[0] == len(indexes)
        assembly = NeuroidAssembly(activations,
                                   coords=coords, dims=['presentation', 'neuroid'])
        model_name = os.path.splitext(os.path.basename(activations_path_pit))[0]
        basenets.append(model_name)
        target_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'output/candidate_models.models.model_activations',
            'model={},stimulus_set=dicarlo.hvm,weights=imagenet,image_size=224,pca_components=1000.pkl'
                .format(model_name)))
        print("-->", target_path)
        with open(target_path, 'wb') as target_file:
            pickle.dump({'data': assembly}, target_file)

    print(" ".join(basenets))


if __name__ == '__main__':
    package()
