"""vit_large_patch14_clip_altahan_02: ViT-L/14 CLIP (OpenAI) fine-tuned on ImageNet-1k with single-step adversarial
training and interpolated halfway back to its starting weights; V4 and IT read through parameter-free grid pools of two
attention blocks.

timm `vit_large_patch14_clip_224.openai_ft_in1k` (the CLIP ViT-L/14 image tower, fine-tuned on ImageNet-1k by timm),
fine-tuned further on ImageNet-1k (AdamW, lr 1e-5, weight decay 0.05, cosine decay, batch 256, 2.56M images,
random-resized-crop and horizontal flip) with single-step adversarial training on every image (FGSM with a random
start, L-inf eps 1/255, step 1.25/255; the loss is the cross-entropy on the perturbed image), after which the weights
are interpolated halfway back to the starting checkpoint (WiSE-FT: 0.5 * starting + 0.5 * fine-tuned). Nothing is
added to the network: the state dict is timm's (304M parameters). Read out with the leaderboard's committed layer map
for the timm model (V1 blocks.6, V2 blocks.7.attn, behaviour fc_norm) except V4 and IT, which are read through
parameter-free pools of the attention outputs: `v4_pool` is blocks.6.attn's 16x16 patch grid adaptive-average-pooled
to 12x12 plus the CLS token (148,480-d), `it_pool` blocks.10.attn's grid pooled to 4x4 plus the CLS token (17,408-d).
forward() is the plain timm forward pass: every layer, including the behavioural readout `fc_norm` (an identity in this
timm model: the class token after the final norm, 1024-d), sees the same image, and forward() returns the 1000
ImageNet logits. The taps are registered before the base network's modules, so Brain-Score's default `logits` layer is
`head`, the output of forward(). Preprocessing is timm's eval transform for this model (resize the short side to 224
with bicubic interpolation, centre-crop 224, CLIP normalisation), not Brain-Score's default.
"""
import numpy as np
import timm
import torch

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'vit_large_patch14_clip_altahan_02'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
WEIGHTS_RELATIVE_PATH = 'user_873/vit_large_patch14_clip_altahan_02.pt'
WEIGHTS_VERSION_ID = 'Yoh3zR7JP7dOinkJAr7QnwgYYwqDOnfd'
WEIGHTS_SHA1 = 'a2017ff65e28675cb21161066b741fbac74c5623'   # checked after download (load_file does not verify it)

BIBTEX = """@inproceedings{radford2021learning,
                  title = {Learning Transferable Visual Models From Natural Language Supervision},
                  author = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
                  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
                  pages = {8748--8763},
                  year = {2021},
                  publisher = {PMLR}
                  }"""

TIMM_NAME = 'vit_large_patch14_clip_224.openai_ft_in1k'

IMAGE_SIZE = 224
MEAN = (0.48145466, 0.4578275, 0.40821073)      # CLIP's normalisation (timm data config), not ImageNet's
STD = (0.26862954, 0.26130258, 0.27577711)
# The 24 transformer blocks are `blocks.<i>`; each has `attn` (attention output) and `mlp`. `fc_norm` is the pooled,
# normalised token the ImageNet head reads. Brain-Score's committed map for this model (timm_models plugin):
LAYERS = ['blocks.6', 'v4_pool', 'blocks.7.attn', 'blocks.10.attn', 'it_pool', 'fc_norm']
# IT is read from `it_pool`, blocks.10.attn's 16x16 patch grid average-pooled to 4x4 plus the CLS token (17,408-d) instead of
# all 257 tokens (263k-d): Brain-Score's PLS on the full tokens took 5.6 h per MajajHong-IT evaluation, the pooled readout
# 21 min, with the root's IT scores within 0.02 (MajajHong 0.404 vs 0.421, Bracci 0.228 vs 0.251, Coggan IT 0.795 vs 0.697).
# V4 is read from `v4_pool`, blocks.6.attn's patch grid adaptive-average-pooled to 12x12 plus the CLS token (148,480-d).
REGION_LAYER_MAP = {'V1': 'blocks.6', 'V2': 'blocks.7.attn', 'V4': 'v4_pool', 'IT': 'it_pool'}
BEHAVIORAL_READOUT_LAYER = 'fc_norm'
VISUAL_DEGREES = 8            # how large the model's field of view is taken to be (Brain-Score's default)


class GridPool(torch.nn.Module):
    """The 16x16 patch grid of a (B, 257, dim) block output average-pooled to size x size, plus the CLS token:
    (16 + 1) x 1024 = 17,408-d readout at size 4 instead of 263k. No parameters."""
    def __init__(self, size=4):
        super().__init__()
        self.size = size

    def forward(self, x):
        cls, patches = x[:, :1], x[:, 1:]
        b, n, d = patches.shape
        g = int(n ** 0.5)
        pooled = torch.nn.functional.adaptive_avg_pool2d(patches.transpose(1, 2).reshape(b, d, g, g), self.size)
        return torch.cat([cls, pooled.flatten(2).transpose(1, 2)], dim=1).flatten(1)


class WithPooledIT(torch.nn.Module):
    """The timm ViT plus the `v4_pool` tap fed by blocks.6.attn and the `it_pool` tap fed by blocks.10.attn. The taps are registered first, so `head` stays the
    last-registered module (where the leaderboard looks for the logits); the base's modules and parameters are
    re-registered under their own names, so state-dict keys match timm's and `.to(device)` moves everything."""

    def __init__(self, base):
        super().__init__()
        self.v4_pool = GridPool(12)
        self.it_pool = GridPool()
        for name, p in base.named_parameters(recurse=False):   # cls_token, pos_embed
            setattr(self, name, p)
        for name, child in base.named_children():
            setattr(self, name, child)
        self.__dict__['_base'] = base                          # not a submodule: no duplicate parameters
        self.blocks[6].attn.register_forward_hook(self._v4_tap)
        self.blocks[10].attn.register_forward_hook(self._tap)

    def _v4_tap(self, module, inputs, output):
        self.v4_pool(output)          # as _tap, for V4

    def _tap(self, module, inputs, output):
        self.it_pool(output)          # runs the tap for its own hook; returns None so the block's output is unchanged

    def forward(self, x):
        return self._base.forward_head(self._base.forward_features(x))


def preprocess(image_filepaths):
    """timm's eval transform for this model (resize the short side to 224 with bicubic interpolation, centre-crop
    224, CLIP normalisation): what the leaderboard plugin applies. Returns a (N, 3, 224, 224) float32 array."""
    from PIL import Image
    from torchvision import transforms as T
    tf = T.Compose([T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(IMAGE_SIZE),
                    T.ToTensor(), T.Normalize(MEAN, STD)])
    return np.stack([tf(Image.open(p).convert('RGB')).numpy() for p in image_filepaths])


def get_model_list():
    return [IDENTIFIER]


def get_model(name):
    assert name == IDENTIFIER
    path = load_file(bucket='brainscore-storage', folder_name='brainscore-vision/models',
                     relative_path=WEIGHTS_RELATIVE_PATH, version_id=WEIGHTS_VERSION_ID)
    if sha1_hash(path) != WEIGHTS_SHA1:
        raise ValueError(f'{path} does not match sha1 {WEIGHTS_SHA1}')
    # The architecture only (no download): the fine-tuned weights come from the file above.
    model = timm.create_model(TIMM_NAME, pretrained=False)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model = WithPooledIT(model).eval()
    # timm's eval transform (preprocess above), not Brain-Score's default resize + ImageNet normalisation
    wrapper = PytorchWrapper(identifier=IDENTIFIER, model=model, preprocessing=preprocess)
    wrapper.image_size = IMAGE_SIZE
    return wrapper


def get_layers(name):
    assert name == IDENTIFIER
    return LAYERS


def get_bibtex(model_identifier):
    return BIBTEX


if __name__ == '__main__':
    check_models.check_base_models(__name__)
