import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_tools.activations.pytorch import load_preprocess_images, PytorchWrapper
from model_tools.brain_transformation import ModelCommitment
from model_tools.check_submission import check_models

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class AACN_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, dk, dv, image_size, kernel_size=3, num_heads=8,
                 inference=False):
        super(AACN_Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv

        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert self.dv % self.num_heads == 0, "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"

        self.padding = (self.kernel_size - 1) // 2

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding).to(device)
        self.kqv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)
        self.attn_out = nn.Conv2d(self.dv, self.dv, 1).to(device)

        # Positional encodings
        self.rel_embeddings_h = nn.Parameter(
            torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True))
        self.rel_embeddings_w = nn.Parameter(
            torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True))
        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        dkh = self.dk // self.num_heads
        dvh = self.dv // self.num_heads
        flatten_hw = lambda x, depth: torch.reshape(x, (batch_size, self.num_heads, height * width, depth))

        # Compute q, k, v
        kqv = self.kqv_conv(x)
        k, q, v = torch.split(kqv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (dkh ** -0.5)

        # After splitting, shape is [batch_size, num_heads, height, width, dkh or dvh]
        k = self.split_heads_2d(k, self.num_heads)
        q = self.split_heads_2d(q, self.num_heads)
        v = self.split_heads_2d(v, self.num_heads)

        # [batch_size, num_heads, height*width, height*width]
        qk = torch.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh).transpose(2, 3))

        qr_h, qr_w = self.relative_logits(q)
        qk += qr_h
        qk += qr_w

        weights = F.softmax(qk, dim=-1)

        if self.inference:
            self.weights = nn.Parameter(weights)

        attn_out = torch.matmul(weights, flatten_hw(v, dvh))
        attn_out = torch.reshape(attn_out, (batch_size, self.num_heads, self.dv // self.num_heads, height, width))
        attn_out = self.combine_heads_2d(attn_out)
        # Project heads
        attn_out = self.attn_out(attn_out)
        return torch.cat((self.conv_out(x), attn_out), dim=1)

    # Split channels into multiple heads.
    def split_heads_2d(self, inputs, num_heads):
        batch_size, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads, height, width, depth // num_heads)
        split_inputs = torch.reshape(inputs, ret_shape)
        return split_inputs

    # Combine heads (inverse of split heads 2d).
    def combine_heads_2d(self, inputs):
        batch_size, num_heads, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads * depth, height, width)
        return torch.reshape(inputs, ret_shape)

    # Compute relative logits for both dimensions.
    def relative_logits(self, q):
        _, num_heads, height, width, dkh = q.size()
        rel_logits_w = self.relative_logits_1d(q, self.rel_embeddings_w, height, width, num_heads, [0, 1, 2, 4, 3, 5])
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.rel_embeddings_h, width, height,
                                               num_heads,
                                               [0, 1, 4, 2, 5, 3])
        return rel_logits_h, rel_logits_w

    # Compute relative logits along one dimension.
    def relative_logits_1d(self, q, rel_k, height, width, num_heads, transpose_mask):
        rel_logits = torch.einsum('bhxyd,md->bxym', q, rel_k)
        # Collapse height and heads
        rel_logits = torch.reshape(rel_logits, (-1, height, width, 2 * width - 1))
        rel_logits = self.rel_to_abs(rel_logits)
        # Shape it
        rel_logits = torch.reshape(rel_logits, (-1, height, width, width))
        # Tile for each head
        rel_logits = torch.unsqueeze(rel_logits, dim=1)
        rel_logits = rel_logits.repeat((1, num_heads, 1, 1, 1))
        # Tile height / width times
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, height, 1, 1))
        # Reshape for adding to the logits.
        rel_logits = rel_logits.permute(transpose_mask)
        rel_logits = torch.reshape(rel_logits, (-1, num_heads, height * width, height * width))
        return rel_logits

    # Converts tensor from relative to absolute indexing.
    def rel_to_abs(self, x):
        # [batch_size, num_heads*height, L, 2L−1]
        batch_size, num_heads, L, _ = x.size()
        # Pad to shift from relative to absolute indexing.
        col_pad = torch.zeros((batch_size, num_heads, L, 1)).to(device)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (batch_size, num_heads, L * 2 * L))
        flat_pad = torch.zeros((batch_size, num_heads, L - 1)).to(device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        # Reshape and slice out the padded elements.
        final_x = torch.reshape(flat_x_padded, (batch_size, num_heads, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


def get_model():
    return CORnet_Z()


# All code below this point:
# Authors: qbilius, mschrimpf (github username)
# Github repo: https://github.com/dicarlolab/CORnet

class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_Z(nn.Module):

    def __init__(self, in_channels, out_channels, img_size, kernel_size=3, stride=1, att=False):
        super().__init__()
        if att is False:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                                  stride=(stride, stride), padding=kernel_size // 2)
        else:
            self.conv = AACN_Layer(in_channels=in_channels, out_channels=out_channels, dk=32, dv=32,
                                   kernel_size=kernel_size, num_heads=8, image_size=img_size, inference=False)

        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


class CORnet_Z(nn.Module):
    def __init__(self):
        super().__init__()

        attention = True
        if attention:
            # replace all 3x3 convolutions with an attention-augmented convolution
            att = [False, True, True, True]
        else:
            att = [False, False, False, False]

        self.layer1 = CORblock_Z(3, 64, kernel_size=7, stride=2, img_size=224, att=att[0])
        self.layer2 = CORblock_Z(64, 128, img_size=224 // 4, att=att[1])
        self.layer3 = CORblock_Z(128, 256, img_size=224 // 8, att=att[2])
        self.layer4 = CORblock_Z(256, 512, img_size=224 // 16, att=att[3])
        self.outputlayer = nn.Sequential(
            (nn.AdaptiveAvgPool2d(1)),
            (Flatten()),
            (nn.Linear(512, 8)),
            (Identity())
        )

        # module = self
        # print(module._modules)

        # weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.outputlayer(x)
        return x


"""
Template module for a base model submission to brain-score
"""

model = get_model()

preprocessing = functools.partial(load_preprocess_images, image_size=224)
activations_model = PytorchWrapper(identifier='cornet_z_aacn_8heads', model=model,
                                   preprocessing=preprocessing)
# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='cornet_z_aacn_8heads', activations_model=activations_model,
                        # specify layers to consider
                        layers=['layer1.conv', 'layer2.conv', 'layer3.conv', 'layer4.conv'])


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ['cornet_z_aacn_8heads']


def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """

    assert name == 'cornet_z_aacn_8heads'

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """
    return ['layer1.conv', 'layer2.conv', 'layer3.conv', 'layer4.conv']


def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """
    return ''


if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
