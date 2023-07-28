"""
Module responsible for data augmentation constants and configuration.
"""

import torch as ch
from torchvision import transforms
try:
    from .audio_functions import audio_transforms
except:
    print('Problem with audio transforms. This is fine if you are just using image models but audio models will not work.')

# lighting transform
# https://git.io/fhBOc
IMAGENET_PCA = {
    'eigval':ch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':ch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}
class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

# Special transforms for ImageNet(s)
TRAIN_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        Lighting(0.05, IMAGENET_PCA['eigval'], 
                      IMAGENET_PCA['eigvec'])
    ])
"""
Standard training data augmentation for ImageNet-scale datasets: Random crop,
Random flip, Color Jitter, and Lighting Transform (see https://git.io/fhBOc)
"""

TEST_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
"""
Standard test data processing (no augmentation) for ImageNet-scale datasets,
Resized to 256x256 then center cropped to 224x224.
"""

# Data Augmentation defaults
TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
"""
Generic training data transform, given image side length does random cropping,
flipping, color jitter, and rotation. Called as, for example,
:meth:`robustness.data_augmentation.TRAIN_TRANSFORMS_DEFAULT(32)` for CIFAR-10.
"""

TEST_TRANSFORMS_DEFAULT = lambda size:transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
"""
Generic test data transform (no augmentation) to complement
:meth:`robustness.data_augmentation.TEST_TRANSFORMS_DEFAULT`, takes in an image
side length.
"""

# Audio Training with raw audio
TRAIN_TRANSFORMS_JSINV3_AUDIO_ONLY = None
# TRAIN_TRANSFORMS_JSINV3_AUDIO_ONLY = audio_transforms.AudioCompose([
#      audio_transforms.AudioToTensor(),
#      audio_transforms.CombineWithRandomDBSNR(low_snr=-10, high_snr=10),
#      audio_transforms.RMSNormalizeForegroundAndBackground(rms_level=0.1),
#      audio_transforms.UnsqueezeAudio(dim=0),
#      ])
"""
Training Data preprocessing for JSINV3 Audio Dataset with raw audio.
The background audioset data is randomly combined with the foreground speech
at a DBSNR level that is randomly chosen between -10-10dB. ALl of the signals are
RMS normalized to 0.1.
"""

# Audio Test with raw audio (include same type of noise in the training data)
TEST_TRANSFORMS_JSINV3_AUDIO_ONLY = None
# TEST_TRANSFORMS_JSINV3_AUDIO_ONLY = audio_transforms.AudioCompose([
#      audio_transforms.AudioToTensor(),
#      audio_transforms.CombineWithRandomDBSNR(low_snr=-10, high_snr=10),
#      audio_transforms.RMSNormalizeForegroundAndBackground(rms_level=0.1),
#      audio_transforms.UnsqueezeAudio(dim=0),
#      ])
"""
Test Data preprocessing for JSINV3 Audio Dataset with raw audio.
The background audioset data is randomly combined with the foreground speech
at a DBSNR level that is randomly chosen between -10-10dB. ALl of the signals are
RMS normalized to 0.1.
"""
