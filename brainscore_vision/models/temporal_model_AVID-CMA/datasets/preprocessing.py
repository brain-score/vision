# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import random
from ..utils.videotransforms import video_transforms, volume_transforms, tensor_transforms


class VideoPrep_MSC_CJ(object):
    def __init__(self,
                 crop=(224, 224),
                 color=(0.4, 0.4, 0.4, 0.2),
                 min_area=0.08,
                 augment=True,
                 normalize=True,
                 totensor=True,
                 num_frames=8,
                 pad_missing=False,
                 ):
        self.crop = crop
        self.augment = augment
        self.num_frames = num_frames
        self.pad_missing = pad_missing
        if normalize:
            assert totensor

        if augment:
            transforms = [
                video_transforms.RandomResizedCrop(crop, scale=(min_area, 1.)),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ColorJitter(*color),
            ]
        else:
            transforms = [
                video_transforms.Resize(int(crop[0]/0.875)),
                video_transforms.CenterCrop(crop),
            ]

        if totensor:
            transforms += [volume_transforms.ClipToTensor()]
            if normalize:
                transforms += [tensor_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = video_transforms.Compose(transforms)

    def __call__(self, frames):
        frames = self.transform(frames)
        if self.pad_missing:
            while True:
                n_missing = self.num_frames - frames.shape[1]
                if n_missing > 0:
                    frames = torch.cat((frames, frames[:, :n_missing]), 1)
                else:
                    break
        return frames


class VideoPrep_Crop_CJ(object):
    def __init__(self,
                 resize=(256, 256),
                 crop=(224, 224),
                 color=(0.4, 0.4, 0.4, 0.2),
                 num_frames=8,
                 pad_missing=False,
                 augment=True,
                 normalize=True,
                 totensor=True,
                 ):
        self.resize = resize
        self.crop = crop
        self.augment = augment
        self.num_frames = num_frames
        self.pad_missing = pad_missing
        if normalize:
            assert totensor


        if augment:
            transforms = [
                video_transforms.Resize(resize),
                video_transforms.RandomCrop(crop),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ColorJitter(*color),
            ]
        else:
            transforms = [
                video_transforms.Resize(resize),
                video_transforms.CenterCrop(crop),
            ]
        if totensor:
            transforms += [volume_transforms.ClipToTensor()]
            if normalize:
                transforms += [tensor_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = video_transforms.Compose(transforms)

    def __call__(self, frames):
        if isinstance(frames[0], list):
            return torch.stack([self(f) for f in frames])
        frames = self.transform(frames)
        if self.pad_missing:
            while True:
                n_missing = self.num_frames - frames.shape[1]
                if n_missing > 0:
                    frames = torch.cat((frames, frames[:, :n_missing]), 1)
                else:
                    break
        return frames


class AudioPrep(object):
    def __init__(self, trim_pad=True, duration=None, missing_as_zero=False, augment=False, to_tensor=False, volume=0.1):
        self.trim_pad = trim_pad
        self.missing_as_zero = missing_as_zero
        self.augment = augment
        self.to_tensor = to_tensor
        self.volume = volume
        if trim_pad:
            assert duration is not None
            self.duration = duration

    def __call__(self, sig, sr, duration=None):
        if duration is None:
            duration = self.duration
        num_frames = int(duration*sr)

        # Check if audio is missing
        if self.missing_as_zero and sig is None:
            sig = np.zeros((1, num_frames), dtype=np.float32)

        # Downmix to mono
        sig = sig.mean(0).astype(np.float32)

        # Trim or pad to constant shape
        if self.trim_pad:
            if sig.shape[0] > num_frames:
                sig = sig[:num_frames]
            elif sig.shape[0] < num_frames:
                n_pad = num_frames - sig.shape[0]
                sig = np.pad(sig, (0, n_pad), mode='constant', constant_values=(0., 0.))

        # Augment by changing volume +/- 10%
        if self.augment:
            sig *= random.uniform(1.-self.volume, 1.+self.volume)

        sig = sig[np.newaxis]
        if self.to_tensor:
            sig = torch.from_numpy(sig)

        return sig, sr


class LogSpectrogram(object):
    def __init__(self, fps, n_fft=512, hop_size=0.005, normalize=False):
        self.inp_fps = fps
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.rate = 1./hop_size
        self.normalize = normalize

        if self.normalize:
            if n_fft == 512 and fps == 24000:
                stats = np.load('datasets/assets/audio-spectDB-24k-513-norm-stats.npz')
            elif n_fft == 256 and fps == 24000:
                stats = np.load('datasets/assets/audio-spectDB-24k-257-norm-stats.npz')
            self.mean, self.std = stats['mean'], stats['std']

    def __call__(self, sig, sr, duration=None):
        import librosa
        hop_length = int(self.hop_size * sr)
        spect = np.abs(librosa.stft(sig[0], n_fft=self.n_fft*2, hop_length=hop_length))**2.
        spect = np.concatenate([spect[:1], spect[1:].reshape(self.n_fft//2, 2, -1).mean(1)], 0)
        if duration is not None:
            num_frames = int(duration * self.rate)
            spect = spect[:, :num_frames]

        spect = librosa.core.power_to_db(spect, top_db=100)
        if self.normalize:
            spect = (spect - self.mean[:, np.newaxis]) / (self.std[:, np.newaxis] + 1e-5)
        spect_tensor = torch.from_numpy(spect)
        spect_tensor = torch.transpose(spect_tensor, 0, 1).unsqueeze(0)
        return spect_tensor, self.rate

