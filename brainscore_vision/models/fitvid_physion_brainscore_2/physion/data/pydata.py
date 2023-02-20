import os
import io
import glob
import h5py
import json
from PIL import Image
import numpy as np
import logging
import torch
from  torch.utils.data import Dataset
from torchvision import transforms

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.ERROR)

class TDWDatasetBase(Dataset):
    def __init__(
        self,
        data_root,
        imsize,
        seq_len,
        state_len,
        random_seq=True,
        debug=False,
        subsample_factor=1,
        seed=0,
        ):
        assert isinstance(data_root, list)
        self.imsize = imsize
        self.seq_len = seq_len
        self.state_len = state_len # not necessarily always used
        assert self.seq_len > self.state_len, 'Sequence length {} must be greater than state length {}'.format(self.seq_len, self.state_len)
        self.random_seq = random_seq # whether sequence should be sampled randomly from whole video or taken from the beginning
        self.debug = debug
        self.subsample_factor = subsample_factor
        self.rng = np.random.RandomState(seed=seed)

        self.hdf5_files = []
        for path in data_root:
            assert '*.hdf5' in path
            files = sorted(glob.glob(path))
            files = [fn for fn in files if 'tfrecords' not in fn]
            self.hdf5_files.extend(files)
            logging.info('Processed {} with {} files'.format(path, len(files)))
        self.N = min(20, len(self.hdf5_files)) if debug else len(self.hdf5_files)
        logging.info('Dataset len: {}'.format(self.N))

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.get_seq(index)

    def get_seq(self, index):
        with h5py.File(self.hdf5_files[index], 'r') as f: # load ith hdf5 file from list
            frames = list(f['frames'])
            target_contacted_zone = False
            for frame in reversed(frames):
                lbl = f['frames'][frame]['labels']['target_contacting_zone'][()]
                if lbl: # as long as one frame touching, label is True
                    target_contacted_zone = True
                    break

            assert len(frames)//self.subsample_factor >= self.seq_len, 'Images must be at least len {}, but are {}'.format(self.seq_len, len(frames)//self.subsample_factor)
            if self.random_seq: # randomly sample sequence of seq_len
                start_idx = self.rng.randint(len(frames)-(self.seq_len*self.subsample_factor)+1)
            else: # get first seq_len # of frames
                start_idx = 0
            end_idx = start_idx + (self.seq_len*self.subsample_factor)
            images = []
            img_transforms = transforms.Compose([
                transforms.Resize((self.imsize, self.imsize)),
                transforms.ToTensor(),
                ])
            for frame in frames[start_idx:end_idx:self.subsample_factor]:
                img = f['frames'][frame]['images']['_img'][()]
                img = Image.open(io.BytesIO(img)) # (256, 256, 3)
                img = img_transforms(img)
                images.append(img)

            images = torch.stack(images, dim=0)
            labels = torch.ones((self.seq_len, 1)) if target_contacted_zone else torch.zeros((self.seq_len, 1)) # Get single label over whole sequence
            stimulus_name = f['static']['stimulus_name'][()]

        sample = {
            'images': images,
            'binary_labels': labels,
            'stimulus_name': stimulus_name,
        }
        return sample

class TDWDataset(TDWDatasetBase):
    def __getitem__(self, index):
        sample = self.get_seq(index)
        images = sample['images'] # (seq_len, 3, D', D')
        input_images = images[:self.state_len]
        label_image = images[self.state_len]
        sample.update({
            'input_images': input_images,
            'label_image': label_image,
            })
        return sample
