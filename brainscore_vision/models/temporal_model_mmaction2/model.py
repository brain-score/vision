import os
import numpy as np

import mmengine
import mmaction
from mmaction.apis import init_recognizer
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose, pseudo_collate

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper


HOME = os.path.join(os.path.dirname(mmaction.__file__), "models")


class MMActionModelWrapper(PytorchWrapper):
    meta = None

    def load_meta(self, path=os.path.join(os.path.dirname(__file__), "mmaction2.csv")):
        if self.meta is None:
            import pandas as pd
            self.meta = pd.read_csv(path)
    
    def __init__(self, model_name, process_output=None, *args, **kwargs):
        self.load_meta()

        _num_frames = None
        num_frames = kwargs.get("num_frames")
        if isinstance(num_frames, (list, tuple)):
            if num_frames[0] == num_frames[1]:
                _num_frames = num_frames
        elif num_frames is not None:
            _num_frames = num_frames

        model_data = self.meta[self.meta['name'] == model_name].iloc[0]  # return a Series
        config = model_data['config']
        checkpoint = model_data['checkpoint']
        config = config.replace("https://github.com/open-mmlab/mmaction2/blob/main/", "")
        config_path = os.path.join(HOME, config)
        config = mmengine.Config.fromfile(config_path)

        test_pipeline_cfg = config.test_pipeline
        # SampleFrames: clip_len x frame_interval (sampling interval) x num_clips
        # change every ThreeCrop and TenCrop to CenterCrop
        for i, pipeline in enumerate(test_pipeline_cfg):
            if pipeline['type'] in ['ThreeCrop', 'TenCrop']:
                test_pipeline_cfg[i] = {'type': 'CenterCrop', 'crop_size': pipeline['crop_size']}
            if pipeline['type'] in ['SampleFrames']:
                test_pipeline_cfg[i].update({"num_clips": 1, 'frame_interval': 1})

        model = init_recognizer(config, checkpoint, device="cpu")
        init_default_scope(model.cfg.get('default_scope', 'mmaction'))
        test_pipeline = Compose(test_pipeline_cfg[3:])

        def transform_video(video):
            imgs = video.to_numpy()
            data = {'imgs': imgs, 'num_clips': 1, 'modality': 'RGB'}
            if _num_frames is not None:
                data['clip_len'] = _num_frames
                assert len(imgs) == _num_frames
            else:
                data['clip_len'] = len(imgs)

            data = test_pipeline(data)
            return data
        
        super().__init__(model_name, model, transform_video, process_output, *args, **kwargs)

    def forward(self, inputs):
        data = pseudo_collate(inputs)
        data["inputs"] = [d.to(self._device) for d in data["inputs"]]
        result = self._model.test_step(data)[0]
        return result
    

def get_model(identifier):
    if identifier == "I3D":
        process_output = None
        inferencer_kwargs = {
            "fps": 12.5,
            "layer_activation_format": {
                "backbone.conv1": "CTHW",  # too large: (C: 64, T: *, H: 128, W: 128)
                **{f"backbone.layer{i}": "CTHW" for i in range(1, 5)},
                "cls_head": "C",
            },
            "num_frames": (5, np.inf),
        }

    if identifier == "I3D-nonlocal":
        process_output = None
        inferencer_kwargs = {
            "fps": 12.5,
            "layer_activation_format": {
                "backbone.conv1": "CTHW",  # too large: (C: 64, T: *, H: 128, W: 128)
                **{f"backbone.layer{i}": "CTHW" for i in range(1, 5)},
                "cls_head": "C",
            },
            "num_frames": (5, np.inf),
        }

    if identifier == "TSM":
        process_output = None
        inferencer_kwargs = {
            "fps": 25,
            "layer_activation_format": {},
        }

    if identifier == "SlowFast":
        process_output = None
        inferencer_kwargs = {
            "fps": 12.5,
            "layer_activation_format": {
                "backbone.slow_path.conv1_lateral": "CTHW",
                **{f"backbone.slow_path.layer{i}_lateral": "CTHW" for i in range(1, 4)},
                "cls_head": "C",
            },
            "num_frames": 32,  # TODO: in fact can be multiple of 4?
        }

    if identifier == "X3D":
        process_output = None
        inferencer_kwargs = {
            "fps": 30,
            "layer_activation_format": {
                "backbone.conv1_t": "CTHW",
                **{f"backbone.layer{i}": "CTHW" for i in range(1, 5)},
                "cls_head": "C",
            },
        }

    if identifier == "TimeSformer":
        inferencer_kwargs = {
            "fps": 8,
            "layer_activation_format": {
                "backbone.patch_embed": "CTHW",
                **{f"backbone.transformer_layers.layers.{i}": "HWTC" for i in range(0, 12)},
                "cls_head": "C",
            },
            "num_frames": 8
        }
        def process_output(layer, layer_name, inputs, output):
            if layer_name == "backbone.patch_embed":
                B = inputs[0].shape[0]
                C = output.shape[-1]
                output = output.reshape(B, -1, 14, 14, C)
            if layer_name.startswith("backbone.transformer_layers.layers."):
                output = output[:, 1:]
                B = output.shape[0]
                C = output.shape[-1]
                output = output.reshape(B, 14, 14, -1, C)
            return output
        
    if identifier in ["VideoSwin-B", "VideoSwin-L"]:

        transformer_layers = {
            **{f"backbone.layers.0.blocks.{i}": "THWC" for i in range(2)},
            **{f"backbone.layers.1.blocks.{i}": "THWC" for i in range(2)},
            **{f"backbone.layers.2.blocks.{i}": "THWC" for i in range(18)},
            **{f"backbone.layers.3.blocks.{i}": "THWC" for i in range(2)},
        }

        inferencer_kwargs = {
            "fps": 12.5,
            "layer_activation_format": {
                "backbone.patch_embed": "CTHW",
                **transformer_layers,
                "cls_head": "C",
            },
        }
        process_output = None

    if identifier == "UniFormer-V1":

        transformer_layers = {
            **{f"backbone.blocks1.{i}": "CTHW" for i in range(5)},
            **{f"backbone.blocks2.{i}": "CTHW" for i in range(8)},
            **{f"backbone.blocks3.{i}": "CTHW" for i in range(20)},
            **{f"backbone.blocks4.{i}": "CTHW" for i in range(7)},
        }

        inferencer_kwargs = {
            "fps": 6.25,
            "layer_activation_format": {
                "backbone.pos_drop": "CTHW",
                **transformer_layers,
                "cls_head": "C",
            },
        }
        process_output = None

    if identifier.startswith("UniFormer-V2"):

        if identifier == "UniFormer-V2-B":
            num_frames = 8
            num_transformer_layers = 12
            img_size = 14
        elif identifier == "UniFormer-V2-L":
            num_frames = 32
            num_transformer_layers = 24
            img_size = 16

        transformer_layers = {
            **{f"backbone.transformer.resblocks.{i}": "HWTC" for i in range(num_transformer_layers)},
        }

        inferencer_kwargs = {
            "fps": 25,
            "layer_activation_format": {
                "backbone.conv1": "CTHW",
                **transformer_layers,
                "backbone": "C",
                "cls_head": "C",
            },
            "num_frames": num_frames
        }
        def process_output(layer, layer_name, inputs, output):
            if layer_name.startswith("backbone.transformer.resblocks."):
                T = inputs[1]
                C = output.shape[-1]
                output = output[1:]  # remove the class token
                output = output.reshape(img_size, img_size, -1, T, C).permute(2, 0, 1, 3, 4)  # BHWTC 
            return output

    model = MMActionModelWrapper(identifier, process_output, **inferencer_kwargs)
    return model