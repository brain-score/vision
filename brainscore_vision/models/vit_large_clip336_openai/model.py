"""
Iter-10 — ViT-L/14 OpenAI-CLIP fine-tuned, 336px + in12k (vit_large_patch14_clip_336.openai_ft_in12k_in1k).

The current #1 board model is vit_large_patch14_clip_224.openai_ft_in1k (0.48). The leaderboard top is
dominated by CLIP-pretrained + ImageNet-FINE-TUNED backbones (CLIP's human-aligned shape bias + a strong
trained classifier -> top behavior). This is the SAME #1 recipe scaled on its native axes: 336px (natively
fine-tuned, so no resolution mismatch) + extra in12k pretrain data. behavioral_readout='fc_norm' = the
pre-classifier pooled feature embedding (rich similarity space for the odd-one-out/confusion behaviors).
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools, timm
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images

IDENTIFIER = 'vit_large_clip336_openai'
TIMM_NAME = 'vit_large_patch14_clip_336.openai_ft_in12k_in1k'
IMAGE_SIZE = 336


def get_model(name):
    assert name == IDENTIFIER
    model = timm.create_model(TIMM_NAME, pretrained=True).eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    return ['blocks.3', 'blocks.7', 'blocks.11', 'blocks.15', 'blocks.19', 'blocks.23']


def get_bibtex(model_identifier):
    return "@article{radford2021clip, title={Learning Transferable Visual Models From Natural Language Supervision}, author={Radford, Alec and others}, year={2021}}"


if __name__ == '__main__':
    check_models.check_base_models(__name__)
