import os
import struct
import numpy as np
from PIL import Image as PILImage

from .base import Stimulus


def lazy_load(f):
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_image") or self._image is None:
            self._image = PILImage.open(self._file)
        return f(self, *args, **kwargs)
    return wrapper


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
    
    def to_img(self):
        if not hasattr(self, "_image"):
            self._image = PILImage.open(self._file)
        return self._image
    
    def to_numpy(self):
        return np.array(self.to_img())


class UnknownImageFormat(Exception):
    pass

# source: https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory
def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    size = os.path.getsize(file_path)

    with open(file_path) as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0])-2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height