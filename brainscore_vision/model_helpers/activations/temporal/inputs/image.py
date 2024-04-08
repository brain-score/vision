import os
import struct
import numpy as np
from PIL import Image as PILImage

from .base import Stimulus


class Image(Stimulus):
    def __init__(self, path: str):
        self._file = path
        self._size = get_image_size(path)

    def copy(self):
        return Image(self._path)
    
    @property
    def size(self):
        return self._size
    
    def resize(self, size):
        img = self.copy()
        img._size = size
        return img
    
    def from_path(path):
        return Image(path)
    
    def to_img(self):
        if not hasattr(self, "_image"):
            self._image = PILImage.open(self._file)
        return self._image
    
    def to_numpy(self):
        return np.array(self.to_img())


def get_image_size(path):
    img = PILImage.open(path)
    size = img.size
    img.close()
    return size