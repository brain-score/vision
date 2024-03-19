import yaml
import os

import torch

from .utils.logger import Logger
from .utils import main_utils
from .datasets import preprocessing

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper


HOME = os.path.dirname(os.path.abspath(__file__))

def get_model(identifier):
    
    if identifier == 'AVID-CMA-Kinetics400':
        cfg_path = os.path.join(HOME, "configs/main/avid-cma/kinetics/InstX-N1024-PosW-N64-Top32.yaml")
        weight_path = "/home/ytang/workspace/data/weights/temporal_model_AVID-CMA/AVID-CMA_Kinetics_InstX-N1024-PosW-N64-Top32_checkpoint.pth.tar"

    elif identifier == 'AVID-CMA-Audioset':
        cfg_path = os.path.join(HOME, "configs/main/avid-cma/audioset/InstX-N1024-PosW-N64-Top32.yaml")
        weight_path = "/home/ytang/workspace/data/weights/temporal_model_AVID-CMA/AVID-CMA_Audioset_InstX-N1024-PosW-N64-Top32_checkpoint.pth.tar"

    elif identifier == 'AVID-Kinetics400':
        cfg_path = os.path.join(HOME, "configs/main/avid/kinetics/Cross-N1024.yaml")
        weight_path = "/home/ytang/workspace/data/weights/temporal_model_AVID-CMA/AVID_Kinetics_Cross-N1024_checkpoint.pth.tar"

    elif identifier == 'AVID-Audioset':
        cfg_path = os.path.join(HOME, "configs/main/avid/audioset/Cross-N1024.yaml")
        weight_path = "/home/ytang/workspace/data/weights/temporal_model_AVID-CMA/AVID_Audioset_Cross-N1024_checkpoint.pth.tar"

    else:
        raise ValueError(f"Unknown model identifier: {identifier}")


    cfg = yaml.safe_load(open(cfg_path))
    cfg['model']['args']['checkpoint'] = weight_path
    logger = Logger()

    # Define model
    model = main_utils.build_model(cfg['model'], logger)
    
    # take only video model
    model = model.video_model

    # Define dataloaders
    db_cfg = cfg['dataset']
    print(db_cfg)

    num_frames = int(db_cfg['video_clip_duration'] * db_cfg['video_fps'])

    _video_transform = preprocessing.VideoPrep_Crop_CJ(
        resize=(256, 256),
        crop=(db_cfg['crop_size'], db_cfg['crop_size']),
        augment=False,
        num_frames=num_frames,
        pad_missing=True,
    )

    def video_transform(video):
        frames = video.to_pil_imgs()
        return _video_transform(frames)
    
    layer_activation_format = {
        'conv1': 'CTHW',
        **{f"conv{i}x": 'CTHW' for i in range(2, 6)},
    }
    
    return PytorchWrapper(identifier, model, video_transform, fps=db_cfg['video_fps'], layer_activation_format=layer_activation_format)