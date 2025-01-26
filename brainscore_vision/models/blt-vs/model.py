import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torchvision import transforms 
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from pretrained import get_model_instance, clear_models_and_aliases, register_model, register_aliases
from PIL import Image

SUBMODULE_SEPARATOR = '.'

LAYERS = ['Retina_5', 'LGN_5', 'V1_5', 'V2_5', 'V3_5', 'V4_5', 'LOC_5', 'logits']


def get_model(model_name='blt_vs', key_or_alias='blt_vs', image_size=224):
    """
    Get a model instance with preprocessing wrapped in a PytorchWrapper.

    Args:
        model_name (str): Identifier for the model.
        key_or_alias (str): Key or alias for the registered model.
        image_size (int): Input image size for preprocessing.

    Returns:
        PytorchWrapper: A wrapper around the model with preprocessing.
    """
    
    clear_models_and_aliases(BLT_VS)

    register_model(
        BLT_VS,
        'blt_vs',
        'https://zenodo.org/records/14223659/files/blt_vs.zip',
        '36d74a367a261e788028c6c9caa7a5675fee48e938a6b86a6c62655b23afaf53'
    )

    register_aliases(BLT_VS, 'blt_vs', 'blt_vs')


    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocessing = functools.partial(load_preprocess_images_sush, image_size=image_size)

    blt_model = get_model_instance(BLT_VS, key_or_alias)
    blt_model.to(device)
    wrapper = PytorchWrapper(identifier=model_name, model=blt_model, preprocessing=preprocessing)

    return wrapper


def load_preprocess_images_sush(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images_sush(images, image_size=image_size, **kwargs)
    return images


def load_images(image_filepaths):
    return [load_image(image_filepath) for image_filepath in image_filepaths]

def preprocess_images_sush(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input_sush(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images

def torchvision_preprocess_sush(normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5)):
    
    
    
    return transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])
    
def torchvision_preprocess_input_sush(image_size, **kwargs):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        torchvision_preprocess_sush(**kwargs),
    ])
    
def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper() \
                and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image
        
        

class BLT_VS(nn.Module):
    """
    BLT_VS model simulates the ventral stream of the visual cortex. See BLT_VS_info.txt for more details on motivation and design.

    Parameters:
    -----------
    timesteps : int
        Number of time steps for the recurrent computation.
    num_classes : int
        Number of output classes for classification.
    add_feats : int
        Additional features to maintain orientation, color, etc.
    lateral_connections : bool
        Whether to include lateral connections.
    topdown_connections : bool
        Whether to include top-down connections.
    skip_connections : bool
        Whether to include skip connections.
    bio_unroll : bool
        Whether to use biological unrolling.
    image_size : int
        Size of the input image (height and width).
    hook_type : str
        What kind of area/timestep hooks to register. Options are 'concat' (concat BU/TD), 'separate', 'None'.
    """

    def __init__(
        self,
        timesteps=12,
        num_classes=565,
        add_feats=100,
        lateral_connections=True,
        topdown_connections=True,
        skip_connections=True,
        bio_unroll=True,
        image_size=224,
        hook_type='None',
    ):
        super(BLT_VS, self).__init__()

        self.timesteps = timesteps
        self.num_classes = num_classes
        self.add_feats = add_feats
        self.lateral_connections = lateral_connections
        self.topdown_connections = topdown_connections
        self.skip_connections = skip_connections
        self.bio_unroll = bio_unroll
        self.image_size = image_size
        self.hook_type = hook_type

        # Define network areas and configurations
        self.areas = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC", "Readout"]

        if image_size == 224:
            self.kernel_sizes = [7, 7, 5, 1, 5, 3, 3, 5]
            self.kernel_sizes_lateral = [0, 0, 5, 5, 5, 5, 5, 0]
        else:
            self.kernel_sizes = [5, 3, 3, 1, 3, 3, 3, 3]
            self.kernel_sizes_lateral = [0, 0, 3, 3, 3, 3, 3, 0]

        self.strides = [2, 2, 2, 1, 1, 1, 2, 2]
        self.paddings = (np.array(self.kernel_sizes) - 1) // 2  # For 'same' padding
        self.channel_sizes = [
            32,
            32,
            576,
            480,
            352,
            256,
            352,
            int(num_classes + add_feats),
        ]

        # Top-down connections configuration
        self.topdown_connections_layers = [
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ]

        # Initialize network layers
        self.connections = nn.ModuleDict()
        for idx in range(len(self.areas) - 1):
            area = self.areas[idx]
            self.connections[area] = BLT_VS_Layer(
                layer_n=idx,
                channel_sizes=self.channel_sizes,
                strides=self.strides,
                kernel_sizes=self.kernel_sizes,
                kernel_sizes_lateral=self.kernel_sizes_lateral,
                paddings=self.paddings,
                lateral_connections=self.lateral_connections
                and (self.kernel_sizes_lateral[idx] > 0),
                topdown_connections=self.topdown_connections
                and self.topdown_connections_layers[idx],
                skip_connections_bu=self.skip_connections and (idx == 5),
                skip_connections_td=self.skip_connections and (idx == 2),
                image_size=image_size,
            )
        self.connections["Readout"] = BLT_VS_Readout(
            layer_n=7,
            channel_sizes=self.channel_sizes,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            num_classes=num_classes,
        )

        # Create nn.identity for each area for each timesteps such that hooks can be registered to acquire bu and td for any area/timestep
        if self.hook_type != 'None':
            for area in self.areas:
                for t in range(timesteps):
                    if self.hook_type == 'concat' and area != 'Readout': # we can't concat for readout
                        setattr(self, f"{area}_{t}", nn.Identity())
                    else:
                        setattr(self, f"{area}_{t}_BU", nn.Identity())
                        setattr(self, f"{area}_{t}_TD", nn.Identity())
            setattr(self, "logits", nn.Identity())

        # Precompute output shapes
        self.output_shapes = self.compute_output_shapes(image_size)

    def compute_output_shapes(self, image_size):
        """
        Compute the output shapes for each area based on the image size.

        Parameters:
        -----------
        image_size : int
            The input image size.

        Returns:
        --------
        output_shapes : list of tuples
            The output height and width for each area.
        """
        output_shapes = []
        height = width = image_size
        for idx in range(len(self.areas)):
            kernel_size = self.kernel_sizes[idx]
            stride = self.strides[idx]
            padding = self.paddings[idx]
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1
            output_shapes.append((int(height), int(width)))
        return output_shapes

    def forward(
        self,
        img_input,
        extract_actvs=False,
        areas=None,
        timesteps=None,
        bu=True,
        td=True,
        concat=False,
    ):
        """
        Forward pass for the BLT_VS model.

        Parameters:
        -----------
        img_input : torch.Tensor
            Input image tensor.
        extract_actvs : bool
            Whether to extract activations.
        areas : list of str
            List of area names to retrieve activations from.
        timesteps : list of int
            List of timesteps to retrieve activations at.
        bu : bool
            Whether to retrieve bottom-up activations.
        td : bool
            Whether to retrieve top-down activations.
        concat : bool
            Whether to concatenate BU and TD activations.

        Returns:
        --------
        If extract_actvs is False:
            readout_output : list of torch.Tensor
                The readout outputs at each timestep.
        If extract_actvs is True:
            (readout_output, activations) : tuple
                readout_output is as above.
                activations is a dict with structure activations[area][timestep] = activation
        """
        # check if input has 4 dims, else add batch dim
        if len(img_input.shape) == 3:
            img_input = img_input.unsqueeze(0) 

        if extract_actvs:
            if areas is None or timesteps is None:
                raise ValueError(
                    "When extract_actvs is True, areas and timesteps must be specified."
                )
            activations = {area: {} for area in areas}
        else:
            activations = None

        readout_output = []
        bu_activations = [None for _ in self.areas]
        td_activations = [None for _ in self.areas]
        batch_size = img_input.size(0)

        if self.bio_unroll:
            # Implement the bio_unroll forward pass
            bu_activations_old = [None for _ in self.areas]
            td_activations_old = [None for _ in self.areas]

            # Initial activation for Retina
            bu_activations_old[0], _ = self.connections["Retina"](bu_input=img_input)
            bu_activations[0] = bu_activations_old[0]

            # Timestep 0 (if 0 is in timesteps)
            t = 0
            activations = self.activation_shenanigans(
                extract_actvs, areas, timesteps, bu, td, concat, batch_size, bu_activations, td_activations, activations, t
            )

            for t in range(1, self.timesteps):
                # For each timestep, update the outputs of the areas
                for idx, area in enumerate(self.areas[1:-1]):
                    # Update only if necessary
                    should_update = any(
                        [
                            bu_activations_old[idx] is not None, # bottom-up connection
                            (bu_activations_old[2] is not None and (idx + 1) == 5), # skip connection bu
                            td_activations_old[idx + 2] is not None, # top-down connection
                            (td_activations_old[5] is not None and (idx + 1) == 2), # skip connection td
                        ]
                    )
                    if should_update:
                        bu_act, td_act = self.connections[area](
                            bu_input=bu_activations_old[idx],
                            bu_l_input=bu_activations_old[idx + 1],
                            td_input=td_activations_old[idx + 2],
                            td_l_input=td_activations_old[idx + 1],
                            bu_skip_input=bu_activations_old[2]
                            if (idx + 1) == 5
                            else None,
                            td_skip_input=td_activations_old[5]
                            if (idx + 1) == 2
                            else None,
                        )
                        bu_activations[idx + 1] = bu_act
                        td_activations[idx + 1] = td_act

                bu_activations_old = bu_activations[:]
                td_activations_old = td_activations[:]

                # Activate readout when LOC output is ready
                if bu_activations_old[-2] is not None:
                    bu_act, td_act = self.connections["Readout"](
                        bu_input=bu_activations_old[-2]
                    )
                    bu_activations_old[-1] = bu_act
                    td_activations_old[-1] = td_act
                    readout_output.append(bu_act)
                    bu_activations[-1] = bu_act
                    td_activations[-1] = td_act

                activations = self.activation_shenanigans(
                    extract_actvs, areas, timesteps, bu, td, concat, batch_size, bu_activations, td_activations, activations, t
                )

        else:
            # Implement the standard forward pass
            bu_activations[0], _ = self.connections["Retina"](bu_input=img_input)
            for idx, area in enumerate(self.areas[1:-1]):
                bu_act, _ = self.connections[area](
                    bu_input=bu_activations[idx],
                    bu_skip_input=bu_activations[2] if idx + 1 == 5 else None,
                )
                bu_activations[idx + 1] = bu_act

            bu_act, td_act = self.connections["Readout"](bu_input=bu_activations[-2])
            bu_activations[-1] = bu_act
            td_activations[-1] = td_act
            readout_output.append(bu_act)

            for idx,area in enumerate(reversed(self.areas[1:-1])):
                _, td_act = self.connections[area](
                    bu_input=bu_activations[-(idx + 2) - 1],
                    td_input=td_activations[-(idx + 2) + 1],
                    td_skip_input=td_activations[5] if idx + 1 == 2 else None,
                )
                td_activations[-(idx + 2)] = td_act
            _, td_act = self.connections["Retina"](
                bu_input=img_input,
                td_input=td_activations[1],
            )
            td_activations[0] = td_act

            t = 0
            activations = self.activation_shenanigans(
                extract_actvs, areas, timesteps, bu, td, concat, batch_size, bu_activations, td_activations, activations, t
            )

            for t in range(1, self.timesteps):
                # For each timestep, compute the activations
                for idx, area in enumerate(self.areas[1:-1]):
                    bu_act, _ = self.connections[area](
                        bu_input=bu_activations[idx],
                        bu_l_input=bu_activations[idx + 1],
                        td_input=td_activations[idx + 2],
                        bu_skip_input=bu_activations[2] if idx + 1 == 5 else None,
                    )
                    bu_activations[idx + 1] = bu_act

                bu_act, td_act = self.connections["Readout"](bu_input=bu_activations[-2])
                bu_activations[-1] = bu_act
                td_activations[-1] = td_act
                readout_output.append(bu_act)

                for idx,area in enumerate(reversed(self.areas[1:-1])):
                    _, td_act = self.connections[area](
                        bu_input=bu_activations[-(idx + 2) - 1],
                        td_input=td_activations[-(idx + 2) + 1],
                        td_l_input=td_activations[-(idx + 2)],
                        td_skip_input=td_activations[5] if idx + 1 == 2 else None,
                    )
                    td_activations[-(idx + 2)] = td_act
                _, td_act = self.connections["Retina"](
                    bu_input=img_input,
                    td_input=td_activations[1],
                    td_l_input=td_activations[0],
                )
                td_activations[0] = td_act

                activations = self.activation_shenanigans(
                    extract_actvs, areas, timesteps, bu, td, concat, batch_size, bu_activations, td_activations, activations, t
                )

        if self.hook_type != 'None':
            _ = self.logits(readout_output[-1])

        if extract_actvs:
            return readout_output, activations
        else:
            return readout_output
        
        
    def activation_shenanigans(
            self, extract_actvs, areas, timesteps, bu, td, concat, batch_size, bu_activations, td_activations, activations, t
    ):
        """
        Helper function to implement activation collection and compute relevant for hook registration.

        Parameters:
        -----------
        extract_actvs : bool
            Whether to extract activations.
        areas : list of str
            List of area names to retrieve activations from.
        timesteps : list of int
            List of timesteps to retrieve activations at.
        bu : bool
            Whether to retrieve bottom-up activations.
        td : bool
            Whether to retrieve top-down activations.
        concat : bool
            Whether to concatenate BU and TD activations.
        batch_size : int
            Batch size of the input data.
        bu_activations : list of torch.Tensor
            List of bottom-up activations.
        td_activations : list of torch.Tensor
            List of top-down activations.
        activations : dict
            Dictionary to store activations.
        t : int
            Current timestep.

        Returns:
        --------
        activations : dict
            Updated activations dictionary.
        """
        if extract_actvs and t in timesteps:
            for idx, area in enumerate(self.areas):
                if area in areas:
                    # If concat is True and area is 'Readout', skip
                    if concat and area == 'Readout':
                        continue
                    activation = self.collect_activation(
                        bu_activations[idx],
                        td_activations[idx],
                        bu,
                        td,
                        concat,
                        idx,
                        batch_size,
                    )
                    activations[area][t] = activation

        if self.hook_type != 'None':
            for idx, area in enumerate(self.areas):
                if self.hook_type == 'concat' and area != 'Readout':
                    _ = getattr(self, f"{area}_{t}")(concat_or_not(bu_activations[idx], td_activations[idx], dim=1))
                elif self.hook_type == 'separate':
                    _ = getattr(self, f"{area}_{t}_BU")(bu_activations[idx])
                    _ = getattr(self, f"{area}_{t}_TD")(td_activations[idx])

        return activations


    def collect_activation(
        self, bu_activation, td_activation, bu_flag, td_flag, concat, area_idx, batch_size
    ):
        """
        Helper function to collect activations, handling None values and concatenation.

        Parameters:
        -----------
        bu_activation : torch.Tensor or None
            Bottom-up activation.
        td_activation : torch.Tensor or None
            Top-down activation.
        bu_flag : bool
            Whether to collect BU activations.
        td_flag : bool
            Whether to collect TD activations.
        concat : bool
            Whether to concatenate BU and TD activations.
        area_idx : int
            Index of the area in self.areas.
        batch_size : int
            Batch size of the input data.

        Returns:
        --------
        activation : torch.Tensor or dict
            The collected activation. If concat is True, returns a single tensor.
            If concat is False, returns a dict with keys 'bu' and/or 'td'.
        """
        device = next(self.parameters()).device  # Get the device of the model

        if concat:
            # Handle None activations
            if bu_activation is None and td_activation is None:
                # Get output shape and channels
                channels = self.channel_sizes[area_idx] * 2  # BU and TD activations concatenated
                height, width = self.output_shapes[area_idx]
                zeros = torch.zeros((batch_size, channels, height, width), device=device)
                return zeros
            if bu_activation is None:
                bu_activation = torch.zeros_like(td_activation)
            if td_activation is None:
                td_activation = torch.zeros_like(bu_activation)
            activation = torch.cat([bu_activation, td_activation], dim=1)
            return activation
        else:
            activation = {}
            if bu_flag:
                if bu_activation is not None:
                    activation['bu'] = bu_activation
                elif td_activation is not None:
                    activation['bu'] = torch.zeros_like(td_activation)
                else:
                    # Create zeros of appropriate shape
                    channels = self.channel_sizes[area_idx]
                    height, width = self.output_shapes[area_idx]
                    activation['bu'] = torch.zeros(
                        (batch_size, channels, height, width), device=device
                    )
            if td_flag:
                if td_activation is not None:
                    activation['td'] = td_activation
                elif bu_activation is not None:
                    activation['td'] = torch.zeros_like(bu_activation)
                else:
                    channels = self.channel_sizes[area_idx]
                    height, width = self.output_shapes[area_idx]
                    activation['td'] = torch.zeros(
                        (batch_size, channels, height, width), device=device
                    )
            return activation


class BLT_VS_Layer(nn.Module):
    """
    A single layer in the BLT_VS model, representing a cortical area.

    Parameters:
    -----------
    layer_n : int
        Layer index.
    channel_sizes : list
        List of channel sizes for each layer.
    strides : list
        List of strides for each layer.
    kernel_sizes : list
        List of kernel sizes for each layer.
    kernel_sizes_lateral : list
        List of lateral kernel sizes for each layer.
    paddings : list
        List of paddings for each layer.
    lateral_connections : bool
        Whether to include lateral connections.
    topdown_connections : bool
        Whether to include top-down connections.
    skip_connections_bu : bool
        Whether to include bottom-up skip connections.
    skip_connections_td : bool
        Whether to include top-down skip connections.
    image_size : int
        Size of the input image (height and width).
    """

    def __init__(
        self,
        layer_n,
        channel_sizes,
        strides,
        kernel_sizes,
        kernel_sizes_lateral,
        paddings,
        lateral_connections=True,
        topdown_connections=True,
        skip_connections_bu=False,
        skip_connections_td=False,
        image_size=224,
    ):
        super(BLT_VS_Layer, self).__init__()

        in_channels = 3 if layer_n == 0 else channel_sizes[layer_n - 1]
        out_channels = channel_sizes[layer_n]

        # Bottom-up convolution
        self.bu_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[layer_n],
            stride=strides[layer_n],
            padding=paddings[layer_n],
        )

        # Lateral connections
        if lateral_connections:
            kernel_size_lateral = kernel_sizes_lateral[layer_n]
            self.bu_l_conv_depthwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size_lateral,
                stride=1,
                padding='same',
                groups=out_channels,
            )
            self.bu_l_conv_pointwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.bu_l_conv_depthwise = NoOpModule()
            self.bu_l_conv_pointwise = NoOpModule()

        # Top-down connections
        if topdown_connections:
            self.td_conv = nn.ConvTranspose2d(
                in_channels=channel_sizes[layer_n + 1],
                out_channels=out_channels,
                kernel_size=kernel_sizes[layer_n + 1],
                stride=strides[layer_n + 1],
                padding=(kernel_sizes[layer_n + 1] - 1) // 2
            )
            if lateral_connections:
                self.td_l_conv_depthwise = nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes_lateral[layer_n],
                    stride=1,
                    padding='same',
                    groups=out_channels,
                )
                self.td_l_conv_pointwise = nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            else:
                self.td_l_conv_depthwise = NoOpModule()
                self.td_l_conv_pointwise = NoOpModule()
        else:
            self.td_conv = NoOpModule()
            self.td_l_conv_depthwise = NoOpModule()
            self.td_l_conv_pointwise = NoOpModule()

        # Skip connections
        if skip_connections_bu:
            self.skip_bu_depthwise = nn.Conv2d(
                in_channels=channel_sizes[2],  # From V1
                out_channels=out_channels,
                kernel_size=7 if image_size == 224 else 5,
                stride=1,
                padding='same',
                groups=np.gcd(channel_sizes[2], out_channels),
            )
            self.skip_bu_pointwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.skip_bu_depthwise = NoOpModule()
            self.skip_bu_pointwise = NoOpModule()

        if skip_connections_td:
            self.skip_td_depthwise = nn.Conv2d(
                in_channels=channel_sizes[5],  # From V4
                out_channels=out_channels,
                kernel_size=3,  # V4 to V1 skip connection
                stride=1,
                padding='same',
                groups=np.gcd(channel_sizes[5], out_channels),
            )
            self.skip_td_pointwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.skip_td_depthwise = NoOpModule()
            self.skip_td_pointwise = NoOpModule()

        self.layer_norm_bu = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.layer_norm_td = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def forward(
        self,
        bu_input,
        bu_l_input=None,
        td_input=None,
        td_l_input=None,
        bu_skip_input=None,
        td_skip_input=None,
    ):
        """
        Forward pass for a single BLT_VS layer.

        Parameters:
        -----------
        bu_input : torch.Tensor or None
            Bottom-up input tensor.
        bu_l_input : torch.Tensor or None
            Bottom-up lateral input tensor.
        td_input : torch.Tensor or None
            Top-down input tensor.
        td_l_input : torch.Tensor or None
            Top-down lateral input tensor.
        bu_skip_input : torch.Tensor or None
            Bottom-up skip connection input.
        td_skip_input : torch.Tensor or None
            Top-down skip connection input.

        Returns:
        --------
        bu_output : torch.Tensor
            Bottom-up output tensor.
        td_output : torch.Tensor
            Top-down output tensor.
        """
        # Process bottom-up input
        bu_processed = self.bu_conv(bu_input) if bu_input is not None else 0

        # Process top-down input
        td_processed = (
            self.td_conv(td_input, output_size=bu_processed.size())
            if td_input is not None
            else 0
        )

        # Process bottom-up lateral input
        bu_l_processed = (
            self.bu_l_conv_pointwise(self.bu_l_conv_depthwise(bu_l_input))
            if bu_l_input is not None
            else 0
        )

        # Process top-down lateral input
        td_l_processed = (
            self.td_l_conv_pointwise(self.td_l_conv_depthwise(td_l_input))
            if td_l_input is not None
            else 0
        )

        # Process skip connections
        skip_bu_processed = (
            self.skip_bu_pointwise(self.skip_bu_depthwise(bu_skip_input))
            if bu_skip_input is not None
            else 0
        )
        skip_td_processed = (
            self.skip_td_pointwise(self.skip_td_depthwise(td_skip_input))
            if td_skip_input is not None
            else 0
        )

        # Compute sums
        bu_drive = bu_processed + bu_l_processed + skip_bu_processed
        bu_mod = bu_processed + skip_bu_processed
        td_drive = td_processed + td_l_processed + skip_td_processed
        td_mod = td_processed + skip_td_processed

        # Compute bottom-up output
        if isinstance(td_mod, torch.Tensor):
            if isinstance(bu_drive, torch.Tensor):
                bu_output = F.relu(bu_drive) * 2 * torch.sigmoid(td_mod)
            else:
                bu_output = torch.zeros_like(td_mod)
        else:
            bu_output = F.relu(bu_drive)

        # Compute top-down output
        if isinstance(bu_mod, torch.Tensor):
            if isinstance(td_drive, torch.Tensor):
                td_output = F.relu(td_drive) * 2 * torch.sigmoid(bu_mod)
            else:
                td_output = torch.zeros_like(bu_mod)
        else:
            td_output = F.relu(td_drive)

        bu_output = self.layer_norm_bu(bu_output)
        td_output = self.layer_norm_td(td_output)

        return bu_output, td_output


class BLT_VS_Readout(nn.Module):
    """
    Readout layer for the BLT_VS model.

    Parameters:
    -----------
    layer_n : int
        Layer index.
    channel_sizes : list
        List of channel sizes for each layer.
    kernel_sizes : list
        List of kernel sizes for each layer.
    strides : list
        List of strides for each layer.
    num_classes : int
        Number of output classes for classification.
    """

    def __init__(self, layer_n, channel_sizes, kernel_sizes, strides, num_classes):
        super(BLT_VS_Readout, self).__init__()

        self.num_classes = num_classes
        in_channels = channel_sizes[layer_n - 1]
        out_channels = channel_sizes[layer_n]

        self.readout_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[layer_n],
            stride=strides[layer_n],
            padding=(kernel_sizes[layer_n] - 1) // 2,
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer_norm_td = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def forward(self, bu_input):
        """
        Forward pass for the Readout layer.

        Parameters:
        -----------
        bu_input : torch.Tensor
            Bottom-up input tensor.

        Returns:
        --------
        output : torch.Tensor
            Class scores for classification.
        td_output : torch.Tensor
            Top-down output tensor.
        """
        output_intermediate = self.readout_conv(bu_input)
        output_pooled = self.global_avg_pool(output_intermediate).view(
            output_intermediate.size(0), -1
        )
        output = output_pooled[
            :, : self.num_classes
        ]  # Only pass classes to softmax and loss
        td_output = self.layer_norm_td(F.relu(output_intermediate))

        return output, td_output


class NoOpModule(nn.Module):
    """
    A no-operation module that returns zero regardless of the input.

    This is used in places where an operation is conditionally skipped.
    """

    def __init__(self):
        super(NoOpModule, self).__init__()

    def forward(self, *args, **kwargs):
        """
        Forward pass that returns zero.

        Returns:
        --------
        Zero tensor or zero value as appropriate.
        """
        return 0
    
def concat_or_not(bu_activation, td_activation, dim=1):
    # If both are None, return None
    if bu_activation is None and td_activation is None:
        return None
    
    # If bu_activation is None, create a tensor of zeros like td_activation
    if bu_activation is None:
        bu_activation = torch.zeros_like(td_activation)
    
    # If td_activation is None, create a tensor of zeros like bu_activation
    if td_activation is None:
        td_activation = torch.zeros_like(bu_activation)
    
    # Concatenate along the specified dimension
    return torch.cat([bu_activation, td_activation], dim=dim)

def get_layers(model_name):
    
    brainscore_layers = LAYERS
    
    return  brainscore_layers  
    
def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    return """@misc{KietzmannLab2024BLT-VS,
                author       = {Sushrut Thorat, KietzmannLab},
                title        = {BLT-VS: A Tool for Brain-Like Transfer in Vision Systems},
                year         = {2024},
                url          = {https://github.com/KietzmannLab/BLT-VS},
                note         = {GitHub repository},
                }""" 
    
if __name__ == '__main__':
    check_models.check_base_models(__name__)
       
     
