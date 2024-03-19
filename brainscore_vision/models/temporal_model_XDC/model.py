import torch
from .r2plus1d_18 import r2plus1d_18

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper


model_urls = {
    "r2plus1d_18_xdc_ig65m_kinetics": "https://github.com/HumamAlwassel/XDC/releases/download/model_weights/r2plus1d_18_xdc_ig65m_kinetics-f24f6ffb.pth",
    "r2plus1d_18_xdc_ig65m_random": "https://github.com/HumamAlwassel/XDC/releases/download/model_weights/r2plus1d_18_xdc_ig65m_random-189d23f4.pth",
    "r2plus1d_18_xdc_audioset": "https://github.com/HumamAlwassel/XDC/releases/download/model_weights/r2plus1d_18_xdc_audioset-f29ffe8f.pth",
    "r2plus1d_18_fs_kinetics": "https://github.com/HumamAlwassel/XDC/releases/download/model_weights/r2plus1d_18_fs_kinetics-622bdad9.pth",
    "r2plus1d_18_fs_imagenet": "https://github.com/HumamAlwassel/XDC/releases/download/model_weights/r2plus1d_18_fs_imagenet-ff446670.pth",
}

def xdc_video_encoder(pretraining='r2plus1d_18_xdc_ig65m_kinetics', progress=False, **kwargs):
    '''Pretrained video encoders as in 
    https://arxiv.org/abs/1911.12667

    Pretrained weights of all layers except the FC classifier layer are loaded. The FC layer 
    (of size 512 x num_classes) is randomly-initialized. Specify the keyword argument 
    `num_classes` based on your application (default is 400).

    Args:
        pretraining (string): The model pretraining type to load. Available pretrainings are
            r2plus1d_18_xdc_ig65m_kinetics: XDC pretrained on IG-Kinetics (default)
            r2plus1d_18_xdc_ig65m_random: XDC pretrained on IG-Random
            r2plus1d_18_xdc_audioset: XDC pretrained on AudioSet
            r2plus1d_18_fs_kinetics: fully-supervised Kinetics-pretrained baseline
            r2plus1d_18_fs_imagenet: fully-supervised ImageNet-pretrained baseline
        progress (bool): If True, displays a progress bar of the download to stderr
    '''
    assert pretraining in model_urls, \
        f'Unrecognized pretraining type. Available pretrainings: {list(model_urls.keys())}'
    
    model = r2plus1d_18(pretrained=False, progress=progress, **kwargs)

    state_dict = torch.hub.load_state_dict_from_url(
        model_urls[pretraining], progress=progress, check_hash=True,
    )

    model.load_state_dict(state_dict, strict=False)

    return model


def get_model(identifier):
    if identifier == "XDC-IG65M-Kinetics":
        model = xdc_video_encoder(pretraining='r2plus1d_18_xdc_ig65m_kinetics')
    elif identifier == "XDC-IG65M-Random":
        model = xdc_video_encoder(pretraining='r2plus1d_18_xdc_ig65m_random')
    elif identifier == "XDC-AudioSet":
        model = xdc_video_encoder(pretraining='r2plus1d_18_xdc_audioset')

    transform_video = None

    return PytorchWrapper(identifier, model, transform_video)