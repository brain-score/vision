from brainscore_vision.model_helpers.check_submission import check_models
import functools
import os
from urllib.request import urlretrieve
import torchvision.models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from pathlib import Path
from brainscore_vision.model_helpers import download_weights
import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale
from collections import OrderedDict

# This is an example implementation for submitting resnet-50 as a pytorch model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from brainscore_vision.model_helpers.check_submission import check_models

from plenoptic.tools.conv import blur_downsample as blurDn
from plenoptic.simulate.canonical_computations.steerable_pyramid_freq import SteerablePyramidFreq

class BaselineV1(nn.Module):
    def __init__(
        self, 
        image_size=224, 
        num_scales = 5,
        num_orientations = 4,
        rectify = True,
        complex_cells = True,
        do_divsive_normalization = False,
        normalization_replace = False,
        normalization_constant = 0.2,
        do_l2_pooling = False,
        l2_pooling_size = 3,
        ds_factor = 1
    ):
        super().__init__()
        self.ds_factor = ds_factor
        self.sp = SteerablePyramidFreq(
            image_shape=(image_size, image_size),
            height=num_scales,
            order=num_orientations - 1,
            is_complex=True,
            downsample=False,
        )
        if rectify:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Identity()
        
        self.complex_cells = complex_cells
        self.do_divisive_normalization = do_divsive_normalization
        self.normalization_replace = normalization_replace
        self.normalization_constant = normalization_constant
        if do_l2_pooling:
            self.pool = nn.LPPool2d(norm_type=2, kernel_size=l2_pooling_size, stride=1)
        else:
            self.pool = nn.Identity()

    def divisive_normalization(self, x):
        B, C, H, W = x.shape
        norms = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x / (norms + self.normalization_constant)
        return x

    def forward(self, x):
        x = self.sp(x)
        out_tensor, _ = self.sp.convert_pyr_to_tensor(x)
        low_pass, high_pass = out_tensor[:, 0].real, out_tensor[:, -1].real
        real_coeffs = self.relu(out_tensor[:, 1:-1].real)
        imag_coeffs = self.relu(out_tensor[:, 1:-1].imag)
        simple_cells = torch.cat(
            [real_coeffs, imag_coeffs],
            dim=1,
        )
        complex_cells = out_tensor[:, 1:-1].abs()

        normalized_simple_cells = self.divisive_normalization(simple_cells)
        normalized_complex_cells = self.divisive_normalization(complex_cells)

        # create responses based on the type of cells 
        if not self.complex_cells:
            normalized_responses = normalized_simple_cells
            responses = simple_cells
        else:
            normalized_responses = torch.cat(
                [normalized_simple_cells, normalized_complex_cells],
                dim=1,
            )
            responses = torch.cat(
                [simple_cells, complex_cells],
                dim=1,
            )
        if self.normalization_replace and self.do_divisive_normalization:
            responses = normalized_responses
            responses = responses[:, :, ::ds_factor, ::ds_factor]
            return self.pool(responses)
        else:
            if self.do_divisive_normalization:
                responses = torch.cat(
                    [responses, normalized_responses],
                    dim=1,
                )
            responses = responses[:, :, ::self.ds_factor, ::self.ds_factor]
            responses = self.pool(responses)
            return responses

    def to(self, device):
        self.sp = self.sp.to(device)
        return self

class BaselineV2(BaselineV1):
    def __init__(
        self, 
        image_size=224, 
        num_scales = 5,
        num_orientations = 4,
        rectify = True,
        complex_cells = True,
        do_divsive_normalization = False,
        normalization_replace = False,
        normalization_constant = 0.2,
        downsample = True,
        scattering_style = False,
        num_scales_2 = 4,
        num_orientations_2 = 4,
        ds_factor = 4,
    ):
        super().__init__(
            image_size=image_size, 
            num_scales=num_scales,
            num_orientations=num_orientations,
            rectify=rectify,
            complex_cells=complex_cells,
            do_divsive_normalization=do_divsive_normalization,
            normalization_replace=normalization_replace,
            normalization_constant=normalization_constant,
        )

        self.ds_factor = ds_factor
        self.scattering_style = scattering_style
        self.downsample = downsample
        image_size_2 = image_size // 2 if downsample else image_size
        self.sp2 = SteerablePyramidFreq(
            image_shape=(image_size_2, image_size_2),
            height=num_scales_2,
            order=num_orientations_2,
            is_complex=True,
            downsample=False,
        )

        self.num_scales = num_scales_2
        self.num_orientations = num_orientations_2
        self.V2 = nn.Identity()

    def forward(self, x):
        x = rgb_to_grayscale(x, num_output_channels=1)
        if not self.scattering_style:
            v1_responses = super().forward(x)
        else:
            x = self.sp(x)
            out_tensor, _ = self.sp.convert_pyr_to_tensor(x)
            real_coeffs = self.relu(out_tensor[:, 1:-1].real)
            imag_coeffs = self.relu(out_tensor[:, 1:-1].imag)
            complex_cells = out_tensor[:, 1:-1].abs()
            v1_responses = complex_cells

        if self.downsample:
            x = blurDn(v1_responses, 1)
        
        v2_responses = self.sp2(x)
        v2_responses, _ = self.sp2.convert_pyr_to_tensor(v2_responses)
        if self.downsample:
            v1_responses = blurDn(v1_responses, 1)
        channels_v2_per_input = self.num_scales * self.num_orientations + 2 
        v2_responses = self.sp2(v1_responses)
        v2_responses, _ = self.sp2.convert_pyr_to_tensor(v2_responses)

        # cut out low/high pass 
        bands = v2_responses[:, ::channels_v2_per_input, :, :]
        bands = [v2_responses[:, i*channels_v2_per_input:(i+1)*channels_v2_per_input, :, :] for i in range(v1_responses.shape[1])]
        bands = [band[:, 1:-1] for band in bands]
        bands = torch.cat(bands, dim=1)

        # apply nonlinearities in same way as V1
        real_coeffs = self.relu(bands.real)
        imag_coeffs = self.relu(bands.imag)

        simple_cells = torch.cat(
            [real_coeffs, imag_coeffs],
            dim=1,
        )
        complex_cells = bands.abs()

        normalized_simple_cells = self.divisive_normalization(simple_cells)
        normalized_complex_cells = self.divisive_normalization(complex_cells)

        # create responses based on the type of cells 
        if not self.complex_cells:
            normalized_responses = normalized_simple_cells
            responses = simple_cells
        else:
            normalized_responses = torch.cat(
                [normalized_simple_cells, normalized_complex_cells],
                dim=1,
            )
            responses = torch.cat(
                [simple_cells, complex_cells],
                dim=1,
            )
        if self.normalization_replace and self.do_divisive_normalization:
            responses = self.V2(normalized_responses[:, :, ::self.ds_factor, ::self.ds_factor])
            return responses
        else:
            if self.do_divisive_normalization:
                responses = torch.cat(
                    [responses, normalized_responses],
                    dim=1,
                )
            responses = self.V2(responses[:, :, ::self.ds_factor, ::self.ds_factor])  
            return responses 

    def to(self, device):
        self.sp = self.sp.to(device)
        self.sp2 = self.sp2.to(device)
        return self

def get_model_list():
    return ["steer_pyr_scatter_v2_ds4"]


def get_model(name):
    assert name == "steer_pyr_scatter_v2_ds4"
    model = BaselineV2(
        num_scales=5,
        num_orientations=4,
        rectify=True,
        complex_cells=True,
        do_divsive_normalization=False,
        normalization_replace=False,
        normalization_constant=0.2,
        scattering_style=True,
        num_scales_2=4,
        num_orientations_2=4,
        downsample=True,
    )
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    wrapper.image_size = 224
    return wrapper

def get_layers(name):
    assert name == "steer_pyr_scatter_v2_ds4"

    outs = ["V2"]
    return outs


def get_bibtex(model_identifier):
    return """xx"""


if __name__ == "__main__":
    check_models.check_base_models(__name__)
