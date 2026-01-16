"""
This file was written to provide an easy way to load an existing CLAPP trained model.
It is based on codebase of CLAPP as modified by Ariane Delrocq.
Import the function load_model from this file. Check its docstring (with help(load_model)) to know how to use.
Note that due to CLAPP training requirements, CLAPP models split their input image into smaller patches, and process the
patches independently. The keyword keep_patches controls whether the representations for each patch are returned,
or averaged together.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

import os
import sys

# clf0 is eval_cur_best_ep300_l6

class Options:
    def __init__(self, option, seed=42, clf=False):
        # arguments for all models
        self.grayscale = True
        self.weight_init = False
        self.kwins_competition = 0.0
        self.kwins_no_soft = False
        self.kwins_no_hard = False
        self.subtract_mean_encodings = False
        self.subtract_mean_encodings_forward = False
        self.subtract_mean_encodings_forward_relu = False
        self.no_detach_mean = False
        self.reduced_patch_pooling = False
        self.l1_kernel_size = 3
        self.no_maxpool = False
        self.no_padding = False
        self.increasing_patch_size = False
        self.no_patch_pooling = False
        self.smaller_vgg = False
        self.model_splits = 6
        self.train_module = [0]
        self.extra_conv = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        # arguments that depend on model
        if option == 0:
            self.patch_size = 27   # change docstring if change this value
            self.no_patch_direction = True
            self.random_patch_location = False
            self.no_patches = False
            self.no_patches_chosen = False
            self.no_patches_chosen_scheme = 0
            self.no_patches_avg_scheme = 0
            self.no_patches_rd_avg_scheme = -1
            self.no_patches_big_region_scheme = -1
            self.no_patches_all2all = False

            if clf:
                self.in_channels = 1024
                self.num_classes = 10
                self.concat = False

        else:
            raise ValueError


def reload_weights(model_path, model, reload_model, device):
    if reload_model:
        print("Loading weights from ", model_path)
        dct = model.match_saved_weights(torch.load(model_path,  map_location=device.type))
        model.load_state_dict(dct)
    else:
        print("Randomly initialized model")
    return model


def reload_clf_weights(clf_path, clf, reload_clf, device):
    if reload_clf:
        print("Loading weights from ", clf_path)
        clf.load_state_dict(torch.load(clf_path, map_location=device.type))
    else:
        print("Randomly initialized model")
    return clf


def _load_model(model_path, opt, reload_model=False):
    model = FullVisionModel(opt)
    model = model.to(opt.device)
    model = reload_weights(model_path, model, reload_model=reload_model, device=opt.device)
    return model


class CenteredActivations(nn.Module):
    def __init__(self, opt, by="channel"):
        """
        by: 'channel', 'chan' to subtract average (over batch) of each channel;
            'position', 'pos' to subtract average (over channels) of each position inf feature map
        """
        super(CenteredActivations, self).__init__()
        self.detach_means = not opt.no_detach_mean
        if "chan" in by:
            self.by = 0
        elif "pos" in by:
            self.by = 1

    def forward(self, x: torch.Tensor):
        # x: b, c, y, x
        means = x.mean(dim=self.by, keepdim=True)
        if self.detach_means:
            means = means.detach()
        return x - means


class KWinnersCompetition(nn.Module):
    """
    Implements the method described by Miconi et al, 2021:
    Applies soft competition to compute activations (mean activation over all neurons substracted to the activations).
    Applies hard k-winners competition for learning: only the k winners undergo plasticity.

    The soft competition:
    input = z, the activations (after potential ReLU)
    computes m = mean(z) (over channels; per location in the feature map)
    returns new activations: z = ReLU( z - m)
    """
    def __init__(self, opt, nb_channels):
        super(KWinnersCompetition, self).__init__()
        self.opt = opt
        if self.opt.kwins_competition >= 1:  # k=kwins_competition for top-k
            self.k = int(self.opt.kwins_competition)
        else:  # kwins_competition is proportion of neurons for top-k
            self.k = int(np.ceil(nb_channels * self.opt.kwins_competition))
        self.relu = nn.ReLU()
        self.apply_soft = self.opt.kwins_no_soft
        self.apply_hard = self.opt.kwins_no_hard
        if self.apply_soft:
            self.center = CenteredActivations(opt, by='pos')

    def forward(self, x: torch.Tensor):
        # x: b, c, y, x
        if self.apply_hard:
            topk_indices = x.topk(self.k, dim=1)[1]
            topk_mask = torch.zeros(x.size(), device=x.device) > 0   # artificially create tensor of filled with False
            topk_mask.scatter_(1, topk_indices, True)   # topk_mask now has True where x is in its top-k, False elsewhere
            non_plastic = x.detach()
            plasticity_masked = torch.where(topk_mask, x, non_plastic)
        else:
            plasticity_masked = x
        if self.apply_soft:
            centered = self.center(plasticity_masked)
            new_x = self.relu(centered)
        else:
            new_x = plasticity_masked
        return new_x


# class ReshapeModule(nn.Module):
#     """
#     This module is only here to reshape the variables being passed from layer to layer in VGG_like_Encoder
#     because the hooks of Brainscore need to recognize the batch elements, put each pacth of each batch element is
#     passed through the model along the batch dimension.
#     """
#     def __init__(self):
#         super(ReshapeModule, self).__init__()
#     def __call__(self, x, n_patches_y, n_patches_x):
#         s1 = x.shape
#         return x.view(-1, n_patches_y, n_patches_x, s1[1], s1[2], s1[3])

class Dummy(nn.Module):
    """
    This class is only here to receive a hook on the value out with the proper shape with patches not in batch dimension
    """
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, x):
        return x



class VGG_like_Encoder(nn.Module):
    def __init__(self, opt, block_idx, blocks, in_channels, patch_size=16, overlap_factor=2):
        super(VGG_like_Encoder, self).__init__()
        self.encoder_num = block_idx
        self.n_blocks = len(blocks)
        self.opt = opt

        # Layer
        self.model = self.make_layers(blocks[block_idx], in_channels, l1_ks=self.opt.l1_kernel_size,
                                      no_maxpool=self.opt.no_maxpool, no_padding=self.opt.no_padding)

        # Params
        self.extra_conv = self.opt.extra_conv

        self.no_patches = self.opt.no_patches
        self.overlap = overlap_factor
        self.increasing_patch_size = self.opt.increasing_patch_size
        if not self.no_patches:
            if self.increasing_patch_size:  # This is experimental... take care, this must be synced with architecture, i.e. number and position of downsampling layers (stride 2, e.g. pooling)
                if self.overlap != 2:
                    raise ValueError("if --increasing_patch_size is true, overlap(_factor) has to be equal 2")
                patch_sizes = [4, 4, 8, 8, 16, 16]
                self.patch_size_eff = patch_sizes[block_idx]
                self.max_patch_size = max(patch_sizes)
                high_level_patch_sizes = [4, 4, 4, 4, 4, 2]
                self.patch_size = high_level_patch_sizes[block_idx]
            else:
                self.patch_size = patch_size

        reduced_patch_pool_sizes = [4, 4, 3, 3, 2, 1]
        if opt.reduced_patch_pooling:
            self.patch_average_pool_out_dim = reduced_patch_pool_sizes[block_idx]
        else:
            self.patch_average_pool_out_dim = 1
        self.no_patch_pooling = opt.no_patch_pooling

        def get_last_index(block):
            if block[-1] == 'M':
                last_ind = -2
            else:
                last_ind = -1
            return last_ind

        last_ind = get_last_index(blocks[block_idx])
        self.in_planes = blocks[block_idx][last_ind]  # nb of channels at output of this block

        # Optional extra conv layer to increase rec. field size
        if self.extra_conv and self.encoder_num < 3:
            self.extra_conv_layer = nn.Conv2d(self.in_planes, self.in_planes, stride=3, kernel_size=3, padding=1)

        if self.opt.weight_init:
            raise NotImplementedError("Weight init not implemented for vgg")
        if self.opt.subtract_mean_encodings:
            self.center_activations = CenteredActivations(self.opt, by="chan")
        else:
            self.center_activations = None

        self.dummy = Dummy()

    def make_layers(self, block, in_channels, batch_norm=False, inplace=False, l1_ks=3, no_maxpool=False,
                    no_padding=False):
        layers = []
        for i, v in enumerate(block):
            if v == 'M':
                if not no_maxpool:
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                if self.encoder_num == 0 and i == 0:
                    ks = l1_ks
                else:
                    ks = 3
                if no_padding:
                    pad = 0
                else:
                    pad = ks // 2
                layers.append(nn.Conv2d(in_channels, v, kernel_size=ks, padding=pad))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=inplace))
                in_channels = v
                if self.opt.subtract_mean_encodings_forward:
                    layers.append(CenteredActivations(self.opt, by="chan"))
                    if self.opt.subtract_mean_encodings_forward_relu:
                        layers.append(nn.ReLU())
                if self.opt.kwins_competition:
                    layers.append(KWinnersCompetition(self.opt, v))
        return nn.Sequential(*layers)

    def forward(self, x, n_patches_y, n_patches_x):
        if self.encoder_num in [0, 2, 4] and not self.no_patches:  # [0,2,4,5]
            if self.encoder_num > 0 and self.increasing_patch_size:
                # undo unfolding of the previous module
                s1 = x.shape
                x = x.reshape(-1, n_patches_y, n_patches_x, s1[1], s1[2], s1[3])  # b, n_patches_y, n_patches_x, c, y, x
                # downsampling to get rid of the overlaps between paches of the previous module
                x = x[:, ::2, ::2, :, :, :]  # b, n_patches_x_red, n_patches_y_red, c, y, x.
                s = x.shape
                x = x.permute(0, 3, 2, 5, 1, 4).reshape(s[0], s[3], s[2], s[5], s[1] * s[4]
                    ).permute(0, 1, 4, 2, 3).reshape(s[0], s[3], s[1] * s[4], s[2] * s[5])  # b, c, Y, X

            if self.encoder_num == 0 or self.increasing_patch_size:
                x = (  # b, c, y, x
                    x.unfold(2, self.patch_size, self.patch_size // self.overlap)  # b, c, n_patches_y, x, patch_size
                    .unfold(3, self.patch_size,
                            self.patch_size // self.overlap)  # b, c, n_patches_y, n_patches_x, patch_size, patch_size
                    .permute(0, 2, 3, 1, 4, 5)  # b, n_patches_y, n_patches_x, c, patch_size, patch_size
                )
                n_patches_y = x.shape[1]
                n_patches_x = x.shape[2]
                x = x.reshape(
                    x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
                )  # b * n_patches_y * n_patches_x, c, patch_size, patch_size

        z = self.model(x)  # b * n_patches_y * n_patches_x, c, y, x

        # Optional extra conv layer with downsampling (stride > 1) here to increase receptive field size ###
        if self.extra_conv and self.encoder_num < 3:
            dec = self.extra_conv_layer(z)
            dec = F.relu(dec, inplace=False)
        else:
            dec = z

        if not self.no_patches and not self.no_patch_pooling:
            # Pool over patch
            # in original CPC/GIM, pooling is done over whole patch, i.e. output shape 1 by 1
            out = F.adaptive_avg_pool2d(dec,
                                        self.patch_average_pool_out_dim)  # b * n_patches_y(_reduced) * n_patches_x(_reduced) (* n_extra_patches * n_extra_patches), c, x_pooled, y_pooled
            # Flatten over channel and pooled patch dimensions x_pooled, y_pooled:
            out = out.reshape(out.shape[0],
                              -1)  # b * n_patches_y(_reduced) * n_patches_x(_reduced) (* n_extra_patches * n_extra_patches),  c * y_pooled * x_pooled
        elif self.no_patches:
            out = dec.permute(0, 2, 3, 1)
            n_patches_x, n_patches_y = out.shape[2], out.shape[1]
            out = out.reshape(out.shape[0] * n_patches_y * n_patches_x, out.shape[3])  # b * y * x, c
        else:
            # Just flatten over channel and space dimensions:
            out = dec.reshape(dec.shape[0], -1)

        n_p_x, n_p_y = n_patches_x, n_patches_y

        out = out.reshape(-1, n_p_y, n_p_x, out.shape[
            1])  # b, n_patches_y, n_patches_x, c * y_pooled * x_pooled OR  b * n_patches_y(_reduced) * n_patches_x(_reduced), n_extra_patches, n_extra_patches, c * y_pooled * x_pooled
        out = out.permute(0, 3, 1,
                          2).contiguous()  # b, c * y_pooled * x_pooled, n_patches_y, n_patches_x  OR  b * n_patches_y(_reduced) * n_patches_x(_reduced), c * y_pooled * x_pooled, n_extra_patches, n_extra_patches
        # for --no_patches, out is: b, c, y, x

        if self.center_activations is not None:
            out = self.center_activations(out)

        xx = self.dummy(out)
        return out, z, n_patches_y, n_patches_x


class FullVisionModel(torch.nn.Module):
    @classmethod
    def match_saved_weights(cls, state_dict):
        new_state_dict = {"encoder.0.model.0.weight": state_dict["l1_w"], "encoder.0.model.0.bias": state_dict["l1_b"],
                          "encoder.1.model.0.weight": state_dict["l2_w"], "encoder.1.model.0.bias": state_dict["l2_b"],
                          "encoder.2.model.0.weight": state_dict["l3_w"], "encoder.2.model.0.bias": state_dict["l3_b"],
                          "encoder.3.model.0.weight": state_dict["l4_w"], "encoder.3.model.0.bias": state_dict["l4_b"],
                          "encoder.4.model.0.weight": state_dict["l5_w"], "encoder.4.model.0.bias": state_dict["l5_b"],
                          "encoder.5.model.0.weight": state_dict["l6_w"], "encoder.5.model.0.bias": state_dict["l6_b"]}
        return new_state_dict

    gray_mean = 0.4120  # values for STL-10, train+unsupervised combined
    gray_std = 0.2570

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.increasing_patch_size = self.opt.increasing_patch_size

        self.encoder = self._create_full_model_vgg(opt)

    def _create_full_model_vgg(self, opt):
        if type(opt.patch_size) == int:
            patch_sizes = [opt.patch_size for _ in range(opt.model_splits)]
        else:
            patch_sizes = opt.patch_size

        arch = [128, 256, 'M', 256, 512, 'M', 1024, 'M', 1024, 'M']
        if opt.model_splits == 1:
            blocks = [arch]
        elif opt.model_splits == 2:
            blocks = [arch[:4], arch[4:]]
        elif opt.model_splits == 4:
            blocks = [arch[:4], arch[4:6], arch[6:8], arch[8:]]
        elif opt.model_splits == 3:
            blocks = [arch[:3], arch[3:6], arch[6:]]
        elif opt.model_splits == 6:
            blocks = [arch[:1], arch[1:3], arch[3:4], arch[4:6], arch[6:8], arch[8:]]
        else:
            raise NotImplementedError

        encoder = nn.ModuleList([])

        if opt.grayscale:
            input_dims = 1
        else:
            input_dims = 3

        for idx, _ in enumerate(blocks):
            if idx == 0:
                in_channels = input_dims
            else:
                if blocks[idx - 1][-1] == 'M':
                    in_channels = blocks[idx - 1][-2]
                else:
                    in_channels = blocks[idx - 1][-1]
            encoder.append(VGG_like_Encoder(opt, idx, blocks, in_channels, patch_size=patch_sizes[idx]))

        return encoder

    def _grayscale_batch(self, x):
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device).view(1, 3, 1, 1)
        return (x * weights).sum(dim=1, keepdim=True)

    def _transform_input(self, x, is_normalized):
        """
        Accepts PIL image, numpy array, or torch tensor.
        If only one image given, prepend batch dimension.
        Applies the necessary transformations to the input: grayscale if given in color;
        normalize if not is_normalized (if color input, should NOT be normalized).
        """
        # convert to tensor
        if isinstance(x, Image.Image):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Unsupported type: {type(x)}")

        x = x.float()

        # Check if single image or batch, and reorder dimensions
        if x.ndim == 3:
            # Single image: (H, W, C) or (C, H, W)
            if x.shape[0] <= 4:
                # Likely (C, H, W)
                x = x.unsqueeze(0)  # (1, C, H, W)
            else:
                # Likely (H, W, C)
                x = x.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        elif x.ndim == 4:
            # Batch: (B, H, W, C) or (B, C, H, W)
            if x.shape[-1] <= 4:
                # (B, H, W, C)
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            # else assume (B, C, H, W)
        else:
            raise ValueError(f"Unsupported shape: {x.shape}")

        if x.max() > 1.0:
            # print("WARNING: supposing values are in 0-255")
            x = x / 255

        if x.size(1) == 1:
            need_grayscale = False
        elif x.size(1) in [3, 4]:
            need_grayscale = True
        else:
            raise RuntimeError("Number of channels in input not recognized: should be 1 for grayscale input and 3"
                               "for color input (will be grayscaled)")

        if need_grayscale:
            x = self._grayscale_batch(x)   # grayscale
        if need_grayscale or not is_normalized:
            x = (x - self.gray_mean) / self.gray_std

        return x

    def forward(self, x, is_normalized=False, all_layers=False, keep_patches=False):
        x = self._transform_input(x, is_normalized)
        n_patches_x, n_patches_y = None, None
        outs = []

        model_input = x
        # forward loop through modules
        for idx, module in enumerate(self.encoder):
            h, z, n_patches_y, n_patches_x = module(model_input, n_patches_y, n_patches_x)
            model_input = z.clone().detach()  # full module output
            if keep_patches:
                outs.append(h)  # out: mean pooled per patch
            else:
                outs.append(h.mean(dim=(2, 3), keepdim=False))
        if all_layers:
            return outs
        return outs[-1]


def load_model(model_path, option=0):
    """
    Loads a CLAPP model with given saved weights.
    Args:
        model_path: path to the directory called 'trained_models' that contains the trained weights for different models.
        option: Only 0 is available for now. This describes the type of CLAPP model to load.
            0 is the current state-of-the-art version, with spatial interpretation, large patch size, and critical periods.

    Returns:
        model: a CLAPP model (VGG-6).
            As a torch.nn.Module, model can be called as such:
                output = model(x, is_normalized=False, all_layers=False, keep_patches=False)
                (keyword arguments optional) where:
                x is a batch of data, in the format of a PIl / numpy array / torch tensor of
                    size (batch_size, 1 or 3, height, width) (with height and width >= 27; recommended: 92)
                    Pass is_normalized=True if x (in 0-1) is already standardized; else pass x raw and
                    is_normalized=False (in this case, mean and std from STL-10 will be used for normalization).
                output is the output representations, of size (batch_size, channels) (default)
                    If model is called with all_layers=True, then output is a list (of length 6) of such a
                    representation at the output of each layer of the network. model(x) = model(x, all_layers=True)[-1]
                    If model is called with keep_patches=True, then the representation(s) are of
                    shape (bs, channels, n_patches_y, n_patches_x) instead of (batch_size, channels). The additional
                    dimensions represent the 'patches' (of size 27x27) that the input image is split into. For
                    keep_patches=False, the representations are the average over all patches. Especially if the input is
                    large, retaining the individual patches may provide richer representations, as it retains some
                    spatial information.
    """
    opt = Options(option=option)

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # choose model
    if option == 0:
        weights_path = hf_hub_download(repo_id="a-del/model_0", filename="CLAPP0.ckpt")
    else:
        raise ValueError

    # load pretrained model
    model = _load_model(weights_path, opt, reload_model=True)
    model.eval()

    if opt.reduced_patch_pooling:
        for module in model.encoder:
            module.patch_average_pool_out_dim = 1

    return model





if __name__ == '__main__':
    here = os.path.dirname(__file__)
    model = load_model(model_path=here, option=0)
    print(model._modules)

