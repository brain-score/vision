import os
import torchvision.datasets as datasets
from IPython.core.debugger import set_trace
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
from pathlib import Path
from PIL import Image

__all__ = ['ImageFolderInstance', 'ImageFolderInstanceSamples']

class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
class ImageFolderInstanceSamples(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __init__(self, n_samples=1, **kwargs):        
        super(ImageFolderInstanceSamples, self).__init__(**kwargs)
        self.n_samples = n_samples
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        
        if self.transform is not None:
            img = [self.transform(img) for i in range(self.n_samples)]

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        target = [target for i in range(self.n_samples)]
        index = [index for i in range(self.n_samples)]
        
        return img, target, index  