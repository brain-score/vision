"""vit_base_patch16_clip_altahan_01_v2: ViT-B/16 CLIP with local response normalisation on the patch embedding, fine-tuned
on ImageNet with a SimCLR auxiliary and framed, edge-map and wide-crop training views; V1, V2 and V4 read from an
anti-aliased patch embedding, IT averaged over half-patch image shifts, behaviour from centred, L2-normalised logits.

timm `vit_base_patch16_clip_224.openai_ft_in12k_in1k` (the CLIP ViT-B/16 image tower, fine-tuned by timm on
ImageNet-12k then ImageNet-1k), with a parameter-free local response normalisation (size 5, alpha 1e-4, beta 0.75,
k 2) across the 768 channels of the patch-embedding output (in place of its identity `patch_embed.norm`). Fine-tuned
further on ImageNet-1k: AdamW, lr 1e-5, weight decay 0.05, batch 256, 2.56M images; cross-entropy plus a SimCLR
NT-Xent auxiliary loss (weight 0.5, temperature 0.2) on the 768-d pre-logits through a projection head that is
discarded after training. Training views: random-resized-crop with area scale 0.35-1 and horizontal flip; for the
cross-entropy input, 10% of the images are replaced by their Sobel edge map, and 80% are then shrunk by a factor
U[0.35, 1] and pasted centred on a uniform grey canvas of random level; the SimCLR views are unframed. The LRN has no
weights, so the state dict is timm's.

Readouts, all parameter-free:
  V1      blocks.2's attention output with the patch embedding anti-aliased (the 16x16 conv evaluated at stride 8,
          then a stride-2 3x3 binomial blur), its 14x14 patch grid average-pooled 2x2 to 7x7, plus the CLS token
  V2      blocks.2's attention output of that anti-aliased pass (all 197 tokens)
  V4      blocks.3's attention output of the same pass
  IT      blocks.6.norm2's patch grid average-pooled to 4x4 plus the CLS token (`it_pool`, 13,056-d), averaged over
          four copies of the image translated by 0 or 8 px in each direction
  behaviour  the 1000 logits, centred and L2-normalised per image
  logits  (label benchmarks) the mean of the logits of the 8 degree image and of its central 7 degrees resized back
The extra passes run inside forward() before the plain pass, and a forward hook keeps a module's last output, so the
timm-named layers and `it_pool` hold the plain pass of the 8 degree image. The readout modules are registered before
the network's own, so Brain-Score's default `logits` layer is `head`, the output of forward(). Preprocessing is timm's
eval transform for this model (resize the short side to 235 with bicubic interpolation, centre-crop 224, CLIP
normalisation), not Brain-Score's default. The model code is shared with variants that switch readouts on through
STEP; at STEP = 14 all of them are on.

Every scored layer comes out flat, (images, features), as it_shift, the behavioural readout and the logits already
were: v1_avgpool, aa_v2 and aa_v4 flatten their (tokens, 768) maps in the order Brain-Score itself flattens them, so
their features are unchanged. Layers of different rank get different neuroid coordinates in Brain-Score, and its
activation cache cannot merge a layer into one of another rank cached for the same images.
"""
import numpy as np
import timm
import torch
import torch.nn.functional as F

from brainscore_core.supported_data_standards.brainio.s3 import load_file, sha1_hash
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models

IDENTIFIER = 'vit_base_patch16_clip_altahan_01_v2'
# The weights file, uploaded through brain-score.org's Large File Upload (Models, vision). Fill in the two values
# from the URL it returns: .../brainscore-vision/models/<WEIGHTS_RELATIVE_PATH>?versionId=<WEIGHTS_VERSION_ID>
# The same file and weights as vit_base_patch16_clip_altahan_01; only the readouts' output shapes differ.
WEIGHTS_RELATIVE_PATH = 'user_873/vit_base_patch16_clip_altahan_01.pt'
WEIGHTS_VERSION_ID = 'V___E8lZoBREyvCr7TRCE0tyYeoq9krz'
WEIGHTS_SHA1 = '5d9f1edd39d6fc5ae9ce85899e09f6b04e4fe313'   # checked after download (load_file does not verify it)

BIBTEX = """@inproceedings{radford2021learning,
                  title = {Learning Transferable Visual Models From Natural Language Supervision},
                  author = {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
                  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
                  pages = {8748--8763},
                  year = {2021},
                  publisher = {PMLR}
                  }"""

STEP = 14
TIMM_NAME = 'vit_base_patch16_clip_224.openai_ft_in12k_in1k'
IMAGE_SIZE = 224
MEAN = (0.48145466, 0.4578275, 0.40821073)      # CLIP's normalisation (timm data config), not ImageNet's
STD = (0.26862954, 0.26130258, 0.27577711)
REGION_LAYER_MAP = {'V1': 'v1_avgpool' if STEP >= 10 else 'blocks.2.attn',
                    'V2': 'aa_v2' if STEP >= 12 else 'blocks.2.attn',
                    'V4': 'aa_v4' if STEP >= 13 else 'blocks.3.attn' if STEP >= 2 else 'blocks.2.attn',
                    'IT': 'it_shift' if STEP >= 14 else 'it_pool'}
BEHAVIORAL_READOUT_LAYER = 'logits_centred_l2' if STEP >= 5 else 'head' if STEP >= 4 else 'fc_norm'
LAYERS = list(dict.fromkeys(['blocks.2.attn', 'blocks.6.norm2', 'it_pool', 'fc_norm', *REGION_LAYER_MAP.values(),
                             BEHAVIORAL_READOUT_LAYER]))
VISUAL_DEGREES = 8            # how large the model's field of view is taken to be (Brain-Score's default)
LABEL_DEGREES = 7             # step 9: forward()'s logits average the VISUAL_DEGREES view with this magnified view
SHIFT = 8                     # step 14: px


class GridPool(torch.nn.Module):
    """The patch grid of a (B, 1 + g*g, dim) token map average-pooled to `size` x `size`, plus the CLS token. No
    parameters. Flattened (it_pool, as in the root: 17 x 768 = 13,056-d; v1_avgpool, 50 x 768 = 38,400-d) or as tokens."""

    def __init__(self, size, flatten):
        super().__init__()
        self.size, self.flatten = size, flatten

    def forward(self, x):
        cls, patches = x[:, :1], x[:, 1:]
        b, n, d = patches.shape
        g = int(n ** 0.5)
        pooled = F.adaptive_avg_pool2d(patches.transpose(1, 2).reshape(b, d, g, g), self.size)
        out = torch.cat([cls, pooled.flatten(2).transpose(1, 2)], dim=1)
        return out.flatten(1) if self.flatten else out


class ChannelLRN(torch.nn.Module):
    """Step 3: LocalResponseNorm(size 5, alpha 1e-4, beta 0.75, k 2) across the channels of (B, tokens, channels)."""

    def __init__(self):
        super().__init__()
        self.lrn = torch.nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)

    def forward(self, x):
        return self.lrn(x.transpose(1, 2)).transpose(1, 2)


class CentredL2(torch.nn.Module):
    """Per-image (l - mean(l)) / ||l - mean(l)||_2 over the last dim. No parameters."""

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class CentralZoom(torch.nn.Module):
    """Keep the central LABEL_DEGREES / VISUAL_DEGREES of the image (196 of 224 px) and resize it back
    (antialiased bilinear). Resizing commutes with the normalization, so this equals zooming the raw image."""

    def forward(self, x):
        h, w = x.shape[-2:]
        ch, cw = round(h * LABEL_DEGREES / VISUAL_DEGREES), round(w * LABEL_DEGREES / VISUAL_DEGREES)
        top, left = (h - ch) // 2, (w - cw) // 2
        x = x[..., top:top + ch, left:left + cw]
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False, antialias=True)


class BinomialBlur(torch.nn.Module):
    """Fixed 3x3 binomial ([1,2,1] outer [1,2,1] / 16) depthwise blur with stride 2 and zero padding 1. No parameters;
    the kernel is a non-persistent buffer, so the state dict is unchanged."""

    def __init__(self):
        super().__init__()
        k = torch.tensor([1., 2., 1.])
        self.register_buffer('kernel', (torch.outer(k, k) / 16)[None, None], persistent=False)

    def forward(self, x):
        return F.conv2d(x, self.kernel.to(x.dtype).expand(x.shape[1], 1, 3, 3), stride=2, padding=1, groups=x.shape[1])


class ViTTaps(torch.nn.Module):
    """The timm ViT plus the taps. The taps are registered first, so `head` stays the last-registered module; the
    base's modules and parameters are re-registered under their own names, so state-dict keys match timm's."""

    def __init__(self, base):
        super().__init__()
        self.it_pool = GridPool(4, flatten=True)          # IT, fed by blocks.6.norm2's hook (as in the root)
        self.zoom = CentralZoom()
        self.blur = BinomialBlur()
        self.v1_avgpool = GridPool(7, flatten=True)       # step 10+: read by its hook only
        self.aa_v2 = torch.nn.Flatten(1)                  # step 12+: blocks.2.attn of the anti-aliased pass, flat
        self.aa_v4 = torch.nn.Flatten(1)                  # step 13+: blocks.3.attn of that pass, flat
        self.it_shift = torch.nn.Identity()               # step 14: it_pool averaged over four shifts
        self.logits_centred_l2 = CentredL2()              # step 5+: read by its hook only
        for name, p in base.named_parameters(recurse=False):   # cls_token, pos_embed
            setattr(self, name, p)
        for name, child in base.named_children():
            setattr(self, name, child)
        self.__dict__['_base'] = base                          # not a submodule: no duplicate parameters
        self.blocks[6].norm2.register_forward_hook(self._tap)

    def _tap(self, module, inputs, output):
        self.it_pool(output)          # runs the tap for its own hook; returns None so the block's output is unchanged

    def tokens(self, patches):
        """Patch tokens -> CLS + position embedding -> norm_pre."""
        b = self._base
        return b.norm_pre(b.patch_drop(b._pos_embed(patches)))

    def head_in(self, x):
        """Blocks' output -> final norm -> CLS token -> fc_norm (the pre-logits)."""
        return self._base.forward_head(self.norm(x), pre_logits=True)

    def pooled(self, x):
        return self.head_in(self.blocks(self.tokens(self.patch_embed(x))))

    @staticmethod
    def attn_out(block, x):
        return block.attn(block.norm1(x))

    def forward(self, x):
        blocks = self.blocks
        if STEP >= 9:
            zoomed = self.pooled(self.zoom(x))
        if STEP >= 11:   # anti-aliased patch embedding: the 16x16 conv at stride 8, then a stride-2 blur (no bias)
            pe = self.patch_embed
            a = self.blur(F.conv2d(x, pe.proj.weight, stride=8)).flatten(2).transpose(1, 2)
            a = blocks[:2](self.tokens(pe.norm(a)))
            v = self.attn_out(blocks[2], a)
            self.v1_avgpool(v)
            if STEP >= 12:
                self.aa_v2(v)
            if STEP >= 13:
                self.aa_v4(self.attn_out(blocks[3], blocks[2](a)))
        if STEP >= 14:
            # IT: it_pool averaged over four copies translated by (0,0), (0,8), (8,0), (8,8) px (reflect-pad the
            # leading edges, crop back); it_pool is linear in its input, so this is the pool of the averaged tokens
            h, w = x.shape[-2:]
            s = torch.cat([F.pad(x, (dx, 0, dy, 0), mode='reflect')[..., :h, :w]
                           for dy, dx in ((0, 0), (0, SHIFT), (SHIFT, 0), (SHIFT, SHIFT))])
            y = blocks[:6](self.tokens(self.patch_embed(s)))
            b6 = blocks[6]
            y = y + b6.drop_path1(b6.ls1(self.attn_out(b6, y)))
            self.it_shift(self.it_pool(b6.norm2(y)).unflatten(0, (4, -1)).mean(0))
        # the unmodified pass last, so the timm-named taps and it_pool hold the 8 deg image
        a = blocks[:2](self.tokens(self.patch_embed(x)))
        if STEP == 10:
            self.v1_avgpool(self.attn_out(blocks[2], a))
        hidden = self.head_in(blocks[2:](a))
        if STEP >= 5:
            self.logits_centred_l2(F.linear(hidden, self.head.weight, self.head.bias))
        # head is linear, so on the mean of the two pre-logit vectors it gives the mean of the two views' logits
        return self.head((hidden + zoomed) / 2) if STEP >= 9 else self.head(hidden)


_TRANSFORM = []


def preprocess(image_filepaths):
    """timm's eval transform for this model, built the way the leaderboard plugin builds it
    (resolve_model_data_config + create_transform: resize the short side to 235 = 224 / 0.95 with bicubic
    interpolation, centre-crop 224, CLIP normalisation). Returns a (N, 3, 224, 224) float32 array."""
    import timm
    from PIL import Image
    if not _TRANSFORM:
        cfg = timm.data.resolve_model_data_config(timm.create_model(TIMM_NAME, pretrained=False))
        _TRANSFORM.append(timm.data.create_transform(**cfg, is_training=False))
    return np.stack([_TRANSFORM[0](Image.open(p).convert('RGB')).numpy() for p in image_filepaths])


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
    if STEP >= 3:   # patch_embed.norm is an Identity in this model; the LRN replaces it (no parameters)
        model.patch_embed.norm = ChannelLRN()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model = ViTTaps(model).eval()
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
