import glob
import os
from shutil import move

def convert_path(old_path, base):
    splits = old_path.split('/')
    post = splits[-3] + '_' + splits[-1]
    return os.path.join(base, post)

old_names = ['clothSagging', 'collision', 'containment', 'dominoes', 'drop', 'linking', 'rollingSliding', 'towers']
new_names = ['Drape', 'Collide', 'Contain', 'Dominoes', 'Drop', 'Link', 'Roll', 'Support']

base_dir = '/mnt/fs4/hsiaoyut/tdw_physics/data'
new_base_dir = '/mnt/fs4/eliwang/physion_data'
for i, name in enumerate(old_names):
    for mode in ['train', 'train_readout', 'valid', 'valid_readout']:
        old_paths = glob.glob(os.path.join(base_dir, name, '*', mode, '*.hdf5'))
        print(name, mode, len(old_paths))
        dst_dir = os.path.join(new_base_dir, mode, new_names[i])
        os.makedirs(dst_dir, exist_ok=True)
        for src in old_paths:
            dst = convert_path(src, dst_dir)
            # print(src, dst)
            move(src, dst)
