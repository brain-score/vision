# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numbers
import random
import numpy as np
import PIL
import torchvision
import warnings
import math

from ..videotransforms import functional as F
from torchvision.transforms import functional as vF

class Compose(object):
    """Composes several transforms

    Args:
    transforms (list of ``Transform`` objects): list of transforms
    to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class RandomDrop(object):
    """Randomly drops frames in order to keep a fixed number of frames
    """

    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, clip):
        assert len(clip) >= self.num_frames
        if len(clip) == self.num_frames:
            return clip
        idx = sorted(random.sample(range(len(clip)), k=self.num_frames))
        return [clip[i] for i in idx]


class UniformDrop(object):
    """Randomly drops frames in order to keep a fixed number of frames
    """

    def __init__(self, num_frames=None, ss_ratio=None):
        self.num_frames = num_frames
        self.ss_ratio = ss_ratio

    def __call__(self, clip):
        if self.num_frames is not None:
            if len(clip) <= self.num_frames:
                return clip
            idx = np.linspace(0, len(clip)-1, self.num_frames, endpoint=True).astype(int)
            return [clip[i] for i in idx]
        elif self.ss_ratio is not None:
            if self.ss_ratio == 1:
                return clip
            idx = np.arange(0, len(clip), self.ss_ratio).astype(int)
            return [clip[i] for i in idx]



class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly
    with a probability 0.5
    """

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip



class RandomGray(object):
    """Horizontally flip the list of given images randomly
    with a probability 0.5
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """

        rand_gray_clip = []
        for img in clip:
            if random.random() < self.p:
                img = vF.to_grayscale(img)
            rand_gray_clip.append(img)
        return rand_gray_clip


class RandomResize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size

    The larger the original image is, the more times it takes to
    interpolate

    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = F.resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size

    The larger the original image is, the more times it takes to
    interpolate

    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = F.resize_clip(
            clip, self.size, interpolation=self.interpolation)
        return resized


class RandomCrop(object):
    """Extract random crop at the same location for a list of images

    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class CenterCrop(object):
    """Extract center crop at the same location for a list of images

    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class TenCrop(object):
    """Extract center crop at the same location for a list of images

    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        cropped = []
        for frame in clip:
            cropped += list(vF.ten_crop(frame, self.size))

        return cropped


class RandomResizedCrop(object):
    """Crops a series of PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, clip):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(clip[0], self.scale, self.ratio)
        return [vF.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in clip]

    def __repr__(self):
        interpolate_str = '?'
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip

    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image

        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError(
                'Color jitter not yet implemented for numpy arrays')
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    img = func(img)
                jittered_clip.append(img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return jittered_clip



class TemporalJitter(object):
    """Crop video sequence temporally with jitter

    Args:
    n_frames (int): number  of output frames
    min_scale (float): minimum subsample rate (default 1.0)
    max_scale (float): maximum subsample rate (default 1.0)
    """

    def __init__(self, n_frames, time_scale=(1.,1.)):
        self.n_frames = n_frames
        self.time_scale = time_scale

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        rate = self.time_scale[0] if self.time_scale[0] == self.time_scale[1] else random.uniform(self.time_scale[0], min(self.time_scale[1], float(len(clip))/float(self.n_frames)))
        clip_ss = [clip[int(t)] for t in np.arange(0, len(clip), rate)]
        if len(clip_ss) == self.n_frames:
            clip_out = clip_ss
        elif len(clip_ss) < self.n_frames:
            # Wrap to fill frames
            clip_out = [clip_ss[t%len(clip_ss)] for t in range(self.n_frames)]
        else:
            # Extract random crop
            max_init_t = len(clip_ss) - self.n_frames
            init_t = random.randint(0, max_init_t)
            clip_out = clip_ss[init_t:init_t + self.n_frames]
        return clip_out



class TemporalCenterCrop(object):
    """Crop video sequence temporally with jitter

    Args:
    n_frames (int): number  of output frames
    min_scale (float): minimum subsample rate (default 1.0)
    max_scale (float): maximum subsample rate (default 1.0)
    """

    def __init__(self, n_frames, time_scale=1.):
        self.n_frames = n_frames
        self.time_scale = time_scale

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        clip_ss = [clip[int(t)] for t in np.arange(0, len(clip), self.time_scale)]
        if len(clip_ss) == self.n_frames:
            clip_out = clip_ss
        elif len(clip_ss) < self.n_frames:
            # Wrap to fill frames
            clip_out = [clip_ss[t%len(clip_ss)] for t in range(self.n_frames)]
        else:
            # Extract random crop
            init_t = (len(clip_ss) - self.n_frames)//2
            clip_out = clip_ss[init_t:init_t + self.n_frames]
        return clip_out