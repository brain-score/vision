import numpy as np
from PIL import Image as PILImage

from .base import Stimulus
from brainscore_vision.model_helpers.activations.temporal.utils import batch_2d_resize


class Image(Stimulus):
    def __init__(self, path: str, size: int):
        self._path = path
        self._size = size

    def copy(self):
        return Image(self._path, self._size)
    
    @property
    def size(self):
        return self._size
    
    def set_size(self, size):
        img = self.copy()
        img._size = size
        return img
    
    def from_path(path):
        return Image(path, get_image_size(path))
    
    def to_pil_img(self):
        return PILImage.fromarray(self.to_numpy())

    def get_frame(self):
        return np.array(PILImage.open(self._path).convert('RGB'))
    
    # return (H, W, C[RGB])
    def to_numpy(self):
        arr = self.get_frame()

        if arr.shape[:2][::-1] != self._size:
            arr = batch_2d_resize(arr[None,:], self._size, "bilinear")[0]

        return arr

    def store_to_path(self, path):
        self.to_img().save(path)
        return path

def get_image_size(path):
    with PILImage.open(path) as img:
        size = img.size
    return size