import h5py
import torch
import glob
try:
    from . import audio_transforms
except:
    print('Problem with audio transforms. This is fine if you are just using image models but audio models will not work.')
import pickle
import numpy as np

class jsinV3_precombined_all_signals(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset. As the authors for information on 
    # datafiles for training.
    hdf5_glob = 'JSIN_all__run_*.h5'
    target_keys = ['signal/word_int', 'signal/speaker_int', 'noise/labels_binary_via_int']

    def __init__(self, root, train=True, download=False, transform=None, eval_max=3):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the
        specified root directory.
        """
        del download

        if train:
            self.all_hdf5_files = glob.glob(root + '/train_*/' + self.hdf5_glob)
        else:
            self.all_hdf5_files = glob.glob(root + '/valid_*/' + self.hdf5_glob)[0:eval_max]

        self.datasets = [H5Dataset(h5_file, transform, self.target_keys) for h5_file in self.all_hdf5_files]

        super().__init__(self.datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map.
        """
        word_and_speaker_encodings = pickle.load( open( "word_and_speaker_encodings_jsinv3.pckl", "rb" ))
        class_map = word_and_speaker_encodings['word_idx_to_word']
        return class_map


class jsinV3_precombined(torch.utils.data.ConcatDataset):
    # Makes a dataset using pre-paired speech and audioset background sounds
    # Works with hdf5 files for the jsinv3 dataset. 
    hdf5_glob = 'JSIN_all__run_*.h5'
    target_keys = ['signal/word_int']

    def __init__(self, root, train=True, download=False, transform=None, eval_max=8):
        """
        Builds the pytorch hdf5 combined dataset from the files found in the 
        specified root directory. 
        """
        del download

        if train:
            self.all_hdf5_files = glob.glob(root + '/train_*/' + self.hdf5_glob)
        else:
            self.all_hdf5_files = glob.glob(root + '/valid_*/' + self.hdf5_glob)[0:eval_max] # Just get one set of them

        self.datasets = [H5Dataset(h5_file, transform, self.target_keys) for h5_file in self.all_hdf5_files]

        super().__init__(self.datasets)

    def class_map(self):
        """
        Loads the mapping between the word IDX and human readable word map. 
        """
        word_and_speaker_encodings = pickle.load( open( "word_and_speaker_encodings_jsinv3.pckl", "rb" )) 
        class_map = word_and_speaker_encodings['word_idx_to_word']
        return class_map


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform, target_keys):
        """
        Builds a pytorch hdf5 dataset
        Args:
            path (str): location of the hdf5 dataset
        """
        self.file_path = path
        self.dataset = None
        self.transform = transform
        self.target_keys = target_keys
        # These TODOs are not implemented for the release. HDF5 files are 
        # already shuffled, so we can run through them directly. 
        # TODO: implement chunking the hdf5 file so that we can shuffle the data
        # TODO: implement shuffling the audioset and the speech separately
        # self.chunk_size = hdf5_chunk_size
        with h5py.File(self.file_path, 'r', swmr=True) as file:
            self.dataset_len = len(file['sources']['signal']['signal'])

    def __getitem__(self, index):
        """
        Gets components of the hdf5 file that are used for training
        Args: 
            index (int): index into the hdf5 file
        Returns:
            [signal, target] : the training audio (signal) containing the preprocessing
              which may combine the foreground and background speech, and the target idx
              specified by target_keys. 
        """
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)# ["ndarray_data"]["signal"]
      
        # Before transforms, set the signal and the noise 
        signal = self.dataset['sources']['signal']['signal'][index]
        noise = self.dataset['sources']['noise']['signal'][index]

        # Transforms will take in the signal and the noise source for this dataset
        # If no transform, just return the speech with no background
        if self.transform is not None:
            signal, noise = self.transform(signal, noise)
        if len(self.target_keys) == 1:
            target_paths = self.target_keys[0].split('/')
            target = self.dataset['sources'][target_paths[0]][target_paths[1]][index]
            if self.target_keys[0] == 'noise/labels_binary_via_int':
                target = target.astype(np.float32)
        # If there are multiple keys, our target has them explicitly listed
        else:
            target = {}
            for target_key in self.target_keys:
                target_paths = target_key.split('/')
                target[target_key] = self.dataset['sources'][target_paths[0]][target_paths[1]][index]
                if target_key == 'noise/labels_binary_via_int':
                    target[target_key] = target[target_key].astype(np.float32)

        return signal, target

    def __len__(self):
        return self.dataset_len
