import os
import imp
import numpy as np
from collections import OrderedDict

import torch
from torchvision import transforms
from openstl.methods import method_maps
from openstl.utils import reshape_patch

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper


# We only use models trained on KITTI dataset, because it is the most ecological, 
# diverse, challenging, and widely used dataset for next frame prediction among 
# the datasets used by OpenSTL repo.
IMAGE_SIZES = (128, 160)  # for KITTI
KITTI_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "kitticaltech")
KITTI_FPS = 10  # BUG: not sure

transform_image = transforms.Resize(IMAGE_SIZES)


class LSTMWrapper(PytorchWrapper):
    def _register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name, target_dict=target_dict):
            output = self._process_activation(_layer, name, _input, output)
            target_dict.setdefault(name, []).append(PytorchWrapper._tensor_to_numpy(output)) 

        hook = layer.register_forward_hook(hook_function)
        return hook
    
    def get_activations(self, inputs, layer_names):
        self._model.eval()
        layer_results = OrderedDict()
        hooks = []

        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            hook = self._register_hook(layer, layer_name, target_dict=layer_results)
            hooks.append(hook)

        with torch.no_grad():
            self.forward(inputs)

        for hook in hooks:
            hook.remove()

        # stack the T dim to be the second dim
        for layer_name, activations in layer_results.items():
            layer_results[layer_name] = np.stack(activations, axis=1)

        return layer_results

    def forward(self, inputs):
        tensor = torch.stack(inputs)
        tensor = tensor.to(self._device)
        return self._model(tensor, return_loss=False)
    

class MIMWrapper(LSTMWrapper):
    def forward(self, inputs):
        output = super().forward(inputs)
        # clear MIMBlock.convlstm_c
        def _clear_helper(module):
            if hasattr(module, "convlstm_c"):
                module.convlstm_c = None
            for child in module.children():
                _clear_helper(child)
        _clear_helper(self._model)
        return output


def _get_config(name, parent_dir):
    config = imp.load_source(name, os.path.join(parent_dir, f"{name}.py")).__dict__
    config = {k: v for k, v in config.items() if not k.startswith("__")}
    return config


def get_model(identifier):
    config = _get_config(identifier, KITTI_CONFIG_DIR)
    config["method"] = config["method"].lower()
    config['dataname'] = "kitticaltech"
    config['dataname'] = "kitticaltech"
    config['metrics'] = ['mse', 'mae']  # not in use, just to initialize the model
    config['in_shape'] = [None, 3, *IMAGE_SIZES]

    if identifier == "PredRNN":
        layer_activation_format = {
            **{f"cell_list.{i}": "TCHW" for i in range(4)},
            "conv_last": "TCHW"
        }

        def process_output(layer, layer_name, inputs, output):
            if layer_name.startswith("cell_list"):
                h, c, m = output
                return m
            else:
                return output
        
        wrapper_cls = LSTMWrapper
        kwargs = {}
        weight_name = "kitticaltech_predrnn_one_ep100.pth"

    elif identifier == "ConvLSTM":
        layer_activation_format = {
            **{f"cell_list.{i}": "TCHW" for i in range(4)},
            "conv_last": "TCHW"
        }

        def process_output(layer, layer_name, inputs, output):
            if layer_name.startswith("cell_list"):
                h, c = output
                return c
            else:
                return output
        
        wrapper_cls = LSTMWrapper
        kwargs = {}
        weight_name = "kitticaltech_convlstm_one_ep100.pth"

    elif identifier in ["SimVP", "TAU"]:
        num_frames = 10
        layer_activation_format = {
            **{f"enc.enc.{i}": "TCHW" for i in range(2)},
            **{f"hid.enc.{i}": "TCHW" for i in range(6)},
            **{f"dec.dec.{i}": "TCHW" for i in range(2)},
        }

        config['in_shape'] = [num_frames, 3, *IMAGE_SIZES]
        wrapper_cls = PytorchWrapper
        kwargs = {"num_frames": num_frames}

        def process_output(layer, layer_name, inputs, output):
            if layer_name.startswith("enc") or layer_name.startswith("dec"):
                output = output.view(-1, num_frames, *output.shape[1:])
            elif layer_name.startswith("hid"):
                output = output[:, None]  # time-compressed layers
            return output
        if identifier == "SimVP":
            weight_name = "kitticaltech_simvp_gsta_one_ep100.pth"
        elif identifier == "TAU":
            weight_name = "kitticaltech_tau_one_ep100.pth"

    elif identifier == "MIM":
        layer_activation_format = {
            **{f"stlstm_layer.{i}": "TCHW" for i in range(4)},
            **{f"stlstm_layer_diff.{i}": "TCHW" for i in range(3)},
            "conv_last": "TCHW"
        }

        def process_output(layer, layer_name, inputs, output):
            if layer_name.startswith("stlstm_layer."):
                h, c, m = output
                ret = m
            elif layer_name.startswith("stlstm_layer_diff."):
                h, c = output
                ret = c
            else:
                ret = output
            return ret
        
        wrapper_cls = MIMWrapper
        kwargs = {}
        weight_name = "kitticaltech_mim_one_ep100.pth"


    model = method_maps[config["method"]](**config).model
    weight_path = f"/home/ytang/workspace/data/weights/temporal_model_openstl/{weight_name}"
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))

    def transform_video_lstm(video):
        frames = torch.Tensor(video.to_numpy() / 255.0).permute(0, 3, 1, 2)
        frames = transform_image(frames)
        frames = frames.permute(0, 2, 3, 1)[None, :]  # BTHWC
        patch_size = config["patch_size"]
        assert 5 == frames.ndim
        batch_size, seq_length, img_height, img_width, num_channels = frames.shape
        a = frames.reshape(batch_size, seq_length,
                                    img_height//patch_size, patch_size,
                                    img_width//patch_size, patch_size,
                                    num_channels)
        b = a.transpose(3, 4)
        patches = b.reshape(batch_size, seq_length,
                                    img_height//patch_size,
                                    img_width//patch_size,
                                    patch_size*patch_size*num_channels)[0]
        return patches
    
    def transform_video_simvp(video):
        frames = torch.Tensor(video.to_numpy() / 255.0).permute(0, 3, 1, 2)
        frames = transform_image(frames)
        return frames
    
    if identifier in ("PredRNN", "ConvLSTM", "MIM"):
        transform_video = transform_video_lstm  
    else: 
        transform_video = transform_video_simvp

    return wrapper_cls(identifier, model, transform_video, fps=KITTI_FPS, 
                          layer_activation_format=layer_activation_format,
                          process_output=process_output, **kwargs)