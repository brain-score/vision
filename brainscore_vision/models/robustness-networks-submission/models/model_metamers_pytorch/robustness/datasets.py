"""
Module containing all the supported datasets, which are subclasses of the
abstract class :class:`robustness.datasets.DataSet`. 

Only includes the Dataset classes that used in Feather et al. 2022. 

Currently supported datasets:

- ImageNet (:class:`robustness.datasets.ImageNet`)
- Word-Speaker-Noise: (:class:`robustness.datasets.jsinV3`)
"""

import os

import torch as ch
import torch.utils.data
from . import imagenet_models
try:
    from . import audio_models
    from .audio_functions import audio_input_representations as air
    from .audio_functions.jsinV3DataLoader_precombined import jsinV3_precombined, jsinV3_precombined_all_signals
    from .audio_functions import audio_transforms as at
except:
    print("Environment not set up for loading audio models")
from torchvision import transforms, datasets

from .tools import constants
from . import data_augmentation as da
from . import loaders

from .tools.helpers import get_label_mapping

import pickle


class DataSet(object):
    '''
    Base class for representing a dataset. Meant to be subclassed, with
    subclasses implementing the `get_model` function. 
    '''

    def __init__(self, ds_name, data_path, **kwargs):
        """
        Args:
            ds_name (str) : string identifier for the dataset
            data_path (str) : path to the dataset 
            num_classes (int) : *required kwarg*, the number of classes in
                the dataset
            mean (ch.tensor) : *required kwarg*, the mean to normalize the
                dataset with (e.g.  :samp:`ch.tensor([0.4914, 0.4822,
                0.4465])` for CIFAR-10)
            std (ch.tensor) : *required kwarg*, the standard deviation to
                normalize the dataset with (e.g. :samp:`ch.tensor([0.2023,
                0.1994, 0.2010])` for CIFAR-10)
            custom_class (type) : *required kwarg*, a
                :samp:`torchvision.models` class corresponding to the
                dataset, if it exists (otherwise :samp:`None`)
            label_mapping (dict[int,str]) : *required kwarg*, a dictionary
                mapping from class numbers to human-interpretable class
                names (can be :samp:`None`)
            transform_train (torchvision.transforms) : *required kwarg*, 
                transforms to apply to the training images from the
                dataset
            transform_test (torchvision.transforms) : *required kwarg*,
                transforms to apply to the validation images from the
                dataset
        """
        required_args = ['num_classes', 'mean', 'std', 'custom_class',
            'label_mapping', 'transform_train', 'transform_test', 'min_value', 'max_value']
        assert set(required_args).issubset(set(kwargs.keys())), "Missing required args, only saw %s" % kwargs.keys()
        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(kwargs)

    def get_model(self, arch, pretrained, arch_kwargs={}):
        '''
        Should be overriden by subclasses. Also, you will probably never
        need to call this function, and should instead by using
        `model_utils.make_and_restore_model </source/robustness.model_utils.html>`_.

        Args:
            arch (str) : name of architecture 
            pretrained (bool): whether to try to load torchvision 
                pretrained checkpoint

        Returns:
            A model with the given architecture that works for each
            dataset (e.g. with the right input/output dimensions).
        '''

        raise NotImplementedError

    def make_loaders(self, workers, batch_size, data_aug=True, subset=None, subset_val=None,
                 subset_start=0, subset_start_val=0, subset_type='rand', subset_type_val='rand', val_batch_size=None,
                 only_val=False, shuffle_train=True, shuffle_val=True,
                 dl_kwargs={}):
        '''
        Args:
            workers (int) : number of workers for data fetching (*required*).
                batch_size (int) : batch size for the data loaders (*required*).
            data_aug (bool) : whether or not to do train data augmentation.
            subset (None|int) : if given, the returned training data loader
                will only use a subset of the training data; this should be a
                number specifying the number of training data points to use.
            subset_start (int) : only used if `subset` is not None; this specifies the
                starting index of the subset.
            subset_type ("rand"|"first"|"last") : only used if `subset is
                not `None`; "rand" selects the subset randomly, "first"
                uses the first `subset` images of the training data, and
                "last" uses the last `subset` images of the training data.
            seed (int) : only used if `subset == "rand"`; allows one to fix
                the random seed used to generate the subset (defaults to 1).
            val_batch_size (None|int) : if not `None`, specifies a
                different batch size for the validation set loader.
            only_val (bool) : If `True`, returns `None` in place of the
                training data loader
            shuffle_train (bool) : Whether or not to shuffle the training data
                in the returned DataLoader.
            shuffle_val (bool) : Whether or not to shuffle the test data in the
                returned DataLoader.
            dl_kwargs (dict): additional keyword arguments for the dataloader

        Returns:
            A training loader and validation loader according to the
            parameters given. These are standard PyTorch data loaders, and
            thus can just be used via:

            >>> train_loader, val_loader = ds.make_loaders(workers=8, batch_size=128) 
            >>> for im, lab in train_loader:
            >>>     # Do stuff...
        '''
        transforms = (self.transform_train, self.transform_test)
        return loaders.make_loaders(workers=workers,
                                    batch_size=batch_size,
                                    transforms=transforms,
                                    data_path=self.data_path,
                                    data_aug=data_aug,
                                    dataset=self.ds_name,
                                    label_mapping=self.label_mapping,
                                    custom_class=self.custom_class,
                                    val_batch_size=val_batch_size,
                                    subset=subset,
                                    subset_val=subset_val,
                                    subset_start=subset_start,
                                    subset_start_val=subset_start_val,
                                    subset_type=subset_type,
                                    subset_type_val=subset_type_val,
                                    only_val=only_val,
                                    shuffle_train=shuffle_train,
                                    shuffle_val=shuffle_val,
                                    dl_kwargs=dl_kwargs)


class ImageNet(DataSet):
    '''
    ImageNet Dataset [DDS+09]_.

    Requires ImageNet in ImageFolder-readable format. 
    ImageNet can be downloaded from http://www.image-net.org. See
    `here <https://pytorch.org/docs/master/torchvision/datasets.html#torchvision.datasets.ImageFolder>`_
    for more information about the format.

    .. [DDS+09] Deng, J., Dong, W., Socher, R., Li, L., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248-255.

    '''
    def __init__(self, data_path, **kwargs):
        """
        """
        mean = kwargs.get('mean', [0.485, 0.456, 0.406])
        std = kwargs.get('std', [0.229, 0.224, 0.225])

        aug_train = kwargs.get('aug_train', da.TRAIN_TRANSFORMS_IMAGENET)
        aug_test = kwargs.get('aug_test', da.TEST_TRANSFORMS_IMAGENET)
        
        ds_kwargs = {
            'num_classes': 1000,
            'mean': ch.tensor(mean),
            'std': ch.tensor(std),
            'min_value': kwargs.get('min_value', 0),
            'max_value': kwargs.get('max_value', 1), 
            'custom_class': None,
            'label_mapping': None,
            'transform_train': aug_train,
            'transform_test': aug_test
        }
        super(ImageNet, self).__init__('imagenet', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained, arch_kwargs={}):
        """
        """
        return imagenet_models.__dict__[arch](num_classes=self.num_classes, 
                                        pretrained=pretrained, **arch_kwargs)


class jsinV3(DataSet):
    """
    Word-Speaker-Noise dataset [Feather et al 2019]

    A set of speech signals that have been precombined with audioset
    background sounds.
    """
    def __init__(self, data_path, 
                 include_rep_in_model=False,
                 audio_representation='mel_spec_0',
                 transform_train=da.TRAIN_TRANSFORMS_JSINV3_AUDIO_ONLY,
                 transform_test=da.TEST_TRANSFORMS_JSINV3_AUDIO_ONLY,
                 include_all_labels=False,
                 include_identity_sequential=False,
                 use_normalization_for_audio_rep=False, # Here to make some old models compatible
                 **kwargs):
        """
        """
        robustness_path = os.path.dirname(os.path.abspath(__file__))
        with open( os.path.join(robustness_path, "audio_functions/word_and_speaker_encodings_jsinv3.pckl"), "rb" ) as f:
            word_and_speaker_encodings = pickle.load(f)
        ds_name = 'jsinV3'
        self.include_rep_in_model=include_rep_in_model
        self.use_normalization_for_audio_rep=use_normalization_for_audio_rep
        self.include_identity_sequential=include_identity_sequential
        self.SR=20000 # Hard coded to match the HDF5 file

        if include_all_labels: # Word, Speaker, and Audioset Labels
            custom_class_jsinV3 = jsinV3_precombined_all_signals
            num_classes = { 
                'signal/word_int': 794,
                'signal/speaker_int': 433,
                'noise/labels_binary_via_int': 517
                }
        else: # Only the word labels
            custom_class_jsinV3 = jsinV3_precombined
            num_classes = 794

        ds_kwargs = {
            'num_classes': num_classes,
            'mean': ch.tensor([0]), # Don't include any normalization of the waveforms
            'std': ch.tensor([1]), # Don't include normalization of the waveforms
            'label_mapping': word_and_speaker_encodings['word_idx_to_word'],
            'custom_class': custom_class_jsinV3,
            'transform_train': transform_train,
            'transform_test': transform_test,
        }

        # Either include the audio representation in the model, or make it be a transformation
        if isinstance(audio_representation, str):
            AUDIO_REP_PROPERTIES = air.AUDIO_INPUT_REPRESENTATIONS[audio_representation]
        else: # Assume that the audio representation is a dictionary that we are parsing
            AUDIO_REP_PROPERTIES = audio_representation
        if include_rep_in_model: 
            if self.use_normalization_for_audio_rep:
                ds_kwargs['audio_kwargs'] = AUDIO_REP_PROPERTIES
            self.AUDIO_REP_PROPERTIES = AUDIO_REP_PROPERTIES
            ds_kwargs['min_value'] = kwargs.get('min_value', -1)
            ds_kwargs['max_value'] =  kwargs.get('max_value', 1)
        else: # Add the transformations to the train and test transforms
            if kwargs.get('audio_rep_on_gpu', False):
                ds_kwargs['audio_rep_transform'] = at.AudioToAudioRepresentation(**AUDIO_REP_PROPERTIES)
                ds_kwargs['transform_train'] = ds_kwargs['transform_train'] 
                ds_kwargs['transform_test'] = ds_kwargs['transform_test'] 
            else:
                ds_kwargs['transform_train'] = at.AudioCompose([
                                                   ds_kwargs['transform_train'], 
                                                   at.AudioToAudioRepresentation(**AUDIO_REP_PROPERTIES),
                                                   ])
                ds_kwargs['transform_test'] = at.AudioCompose([
                                                   ds_kwargs['transform_test'],
                                                   at.AudioToAudioRepresentation(**AUDIO_REP_PROPERTIES),
                                                   ])
            ds_kwargs['max_value'] = kwargs.get('max_value', 100)
            if AUDIO_REP_PROPERTIES['rep_type'] == 'cochleagram':
                ds_kwargs['min_value'] = kwargs.get('min_value', 0)
            else: # spectrogram, do not bound at 0. 
                ds_kwargs['min_value'] = kwargs.get('min_value', -100)
        
        super(jsinV3, self).__init__('jsinV3', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False, arch_kwargs={}):
        """
        """
        if pretrained:
            raise ValueError('jsinV3 does not support pytorch_pretrained=True')
        if self.include_rep_in_model and (not self.use_normalization_for_audio_rep):
            return audio_models.custom_modules.SequentialAttacker(
                       audio_models.custom_modules.AudioInputRepresentation(**self.AUDIO_REP_PROPERTIES),
                       audio_models.__dict__[arch](num_classes=self.num_classes)) 
        # necessary for making adversarial wav examples on older models with normalization training, or 
        # for making adversarial coch examples for model trained with the sequential attacker above. 
        elif self.include_identity_sequential:
            return audio_models.custom_modules.SequentialAttacker(
                       audio_models.custom_modules.SequentialAttacker(), # behaves like identity so state_dict has correct names
                       audio_models.__dict__[arch](num_classes=self.num_classes))
        else:
            return audio_models.__dict__[arch](num_classes=self.num_classes, **arch_kwargs)

DATASETS = {
    'imagenet': ImageNet,
#     'jsinV3': jsinV3,
}
'''
Dictionary of datasets. A dataset class can be accessed as:

>>> import robustness.datasets
>>> ds = datasets.DATASETS['imagenet']('/path/to/imagenet')
'''
