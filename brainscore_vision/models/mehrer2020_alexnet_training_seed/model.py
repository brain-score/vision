import functools

import torch
import torch.nn as nn

from brainscore_core.supported_data_standards.brainio.s3 import load_weight_file
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models

# seed: (classes, S3 version id, sha1); seeds 01-05 are ImageNet-trained, 06-10 ecoset-trained
WEIGHTS = {
    '01': (1000, '8Q4WCKL8YJpnm_lL0R6kClTL4WQMoTc1', '8d506ddba28237340b0d3c163a3de4cca7ca9747'),
    '02': (1000, 'xw.HFPM1NxXaVJD63Vi476BmCoMk15P_', 'bc7a55bba943a37a782b881e3acd67b24864cdff'),
    '03': (1000, 'gODmuM1XzeQrCFAbvvgQT3rlqHhRjnaM', '9686a3dad22f3f161b89bf8145c1f6e442eb7fde'),
    '04': (1000, 'HMaj4PNPrlFa901bRS2ft84NSWMT4Hxh', 'b8aaec2784b0705ad1b01400025d1bdb45d2d689'),
    '05': (1000, 'yQ0AozF7YxX19sO7z54vcEaBHp8q8bOj', '801ea0c0c2833615de7466c3a4d891583dcd62d8'),
    '06': (565, '_2vFin9Bj_azddUxd8Icw7QhfErFKyNE', '247ff566c9a74f942866bbe05b667417fe4543ac'),
    '07': (565, 'IJPGp0um3_j4TCfRDTzsg3XnIsk2r7Wd', '4d9a8333ce1d12f5076b8194079cd3ba101ce9c4'),
    '08': (565, 'bNlZCzWbj_58jzMzhRikYm6xi0_E_spf', '236e17195b67f179e8cc35bfdb1cf835ebd9425c'),
    '09': (565, 'qQ6Lw_kdjN8i4nVTdOwXP15fIr9Ch0lm', '26162e8817e3a46bc1248281ab542e6f2caeb8ae'),
    '10': (565, 'Ih5Jniuxfr5llzxrNZnC6Y4ciWRQjYrL', '38db0aacb6d47297d50c34c25ef3b8b028e03f00'),
}
LAYERS = ['features.0', 'features.3', 'features.6', 'features.8', 'features.10', 'classifier.0', 'classifier.3']


class AlexNetV2(nn.Module):
    """PyTorch port of TF-slim alexnet_v2, the architecture of Mehrer et al. (2020)."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 4096, kernel_size=5), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Conv2d(4096, 4096, kernel_size=1), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Conv2d(4096, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return torch.flatten(self.classifier(self.features(x)), start_dim=1)


def seed_of(name):
    return name.split('training_seed_')[1][:2]


def get_model(name):
    classes, version_id, sha1 = WEIGHTS[seed_of(name)]
    model = AlexNetV2(num_classes=classes)
    weights_path = load_weight_file(bucket="brainscore-storage", folder_name="brainscore-vision/models",
                                    relative_path=f"mehrer2020_alexnet/training_seed_{seed_of(name)}.pth",
                                    version_id=version_id, sha1=sha1)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    # trained on inputs scaled to [-1, 1]
    preprocessing = functools.partial(load_preprocess_images, image_size=224,
                                      normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    return PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)


def get_layers(name):
    return LAYERS


def get_bibtex(model_identifier):
    return """@article{Mehrer_2020,
  title={Individual differences among deep neural network models},
  author={Mehrer, Johannes and Spoerer, Courtney J. and Kriegeskorte, Nikolaus and Kietzmann, Tim C.},
  journal={Nature Communications},
  volume={11},
  number={1},
  year={2020},
  doi={10.1038/s41467-020-19632-w}
}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
