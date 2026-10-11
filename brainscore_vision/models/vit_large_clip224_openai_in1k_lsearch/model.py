"""
Brain-Score submission: OpenAI CLIP ViT-L/14 + ImageNet-1k fine-tune, 224px, auto layer selection.

Strategy notes: this is the same backbone family as the current #1 board model
(vit_large_patch14_clip_224.openai_ft_in1k, 0.48): OpenAI-CLIP's human-aligned shape bias plus a
strongly-trained ImageNet classifier gives the best behavior + robustness (ImageNet-C 0.72)
combination on the board. This submission re-runs that model with an EXPANDED candidate layer list
and no fixed region-layer map, so Brain-Score's built-in LayerSelection cross-validates every
candidate layer (whole-block outputs as well as attention sub-module activations across depth 3-21)
per brain region on the standard region benchmarks (FreemanZiemba2013public V1/V2-pls,
MajajHong2015public V4/IT-pls) and commits to the argmax layer per region. The expanded search
space includes the layer commitments known to work for this family. Behavioral readout = fc_norm
(pooled pre-classifier CLIP embedding; the proven choice for odd-one-out / confusion similarity).
Compute: ViT-L/14 @ 224px (~81 GMACs), comfortably within the fully-scored envelope.
"""
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import timm
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper, load_preprocess_images

IDENTIFIER = 'vit_large_clip224_openai_in1k_lsearch'
TIMM_NAME = 'vit_large_patch14_clip_224.openai_ft_in1k'
IMAGE_SIZE = 224


def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    model = timm.create_model(TIMM_NAME, pretrained=True)
    model.eval()
    preprocessing = functools.partial(load_preprocess_images, image_size=IMAGE_SIZE)
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocessing)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    return [
        # whole-block outputs across depth (candidate pool for region auto-selection)
        'blocks.3', 'blocks.5', 'blocks.6', 'blocks.7', 'blocks.9', 'blocks.11',
        'blocks.13', 'blocks.15', 'blocks.17', 'blocks.19',
        # attention sub-module activations (fine-grained candidates)
        'blocks.6.attn', 'blocks.7.attn', 'blocks.8.attn', 'blocks.10.attn',
        'blocks.12.attn', 'blocks.6.attn.qkv', 'blocks.10.attn.qkv',
        # pooled embedding for behavioral readout
        'fc_norm',
    ]


def get_bibtex(model_identifier):
    return """@article{radford2021clip,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
