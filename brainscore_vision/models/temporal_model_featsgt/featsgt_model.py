from einops import rearrange

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import MaskFormerForInstanceSegmentation

import torch
import torch.nn as nn
import os

import phys_extractors.models.raft_core.raft as raft


def patchify(x, tubelet_size, patch_size):
    '''
    :param x: [B, C, T, H, W]
    :param tubelet_size: 2
    :param patch_size: (8, 8)
    :return:
    '''
    videos_squeeze = rearrange(x,
                               'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                               p0=tubelet_size,
                               p1=patch_size[0],
                               p2=patch_size[1])

    videos_patch = rearrange(videos_squeeze, 'b n p c -> b n (p c)')

    return videos_patch


class FeatsEncoder(nn.Module):
    def __init__(self, model_name, embed_dim=256):
        super().__init__()

        self.mean = torch.tensor([[0.485, 0.456, 0.406]])[:, :, None, None, None]

        self.std = torch.tensor([[0.229, 0.224, 0.225]])[:, :, None, None, None]

        self.embed_dim = embed_dim

        self.pretrained_raft_model = raft.load_raft_model(
            load_path=os.path.expanduser(model_name),
            multiframe=True,
            scale_inputs=True)

        # set requires_grad false
        for param in self.pretrained_raft_model.parameters():
            param.requires_grad = False

        self.depth_model = AutoModelForDepthEstimation.from_pretrained("nielsr/depth-anything-small")

        # set requires_grad false
        for param in self.depth_model.parameters():
            param.requires_grad = False

        # load MaskFormer fine-tuned on COCO panoptic segmentation
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
        self.seg_model = MaskFormerForInstanceSegmentation.from_pretrained(
            "facebook/maskformer-swin-base-coco")  # .cuda()

        # set requires_grad false
        for param in self.seg_model.parameters():
            param.requires_grad = False

        n_flow = 1920
        n_depth = 6144
        n_seg = 4096
        pe_len = 2280

        self.pos_embedding = nn.Parameter(torch.randn(1, pe_len, embed_dim))

    def get_embed_dim(self):
        return self.embed_dim

    def get_num_heads(self):
        return 16

    def get_pos_embed(self):
        return None

    def apply_pos_embed(self, outputs, pos_embed):
        return outputs

    def unnormalize(self, x):
        x = x.transpose(1,2)
        x = x * self.std.to(x.device) + self.mean.to(x.device)
        return x.transpose(1,2)

    def get_flow(self, videos):
        '''
        videos: [B, 3, 16, 224, 224]
        return features of shape: [B, N_patches, D]
        '''

        videos = self.unnormalize(videos)

        with torch.no_grad():
            flow, features = self.pretrained_raft_model(videos)

            # Create a zero tensor with the same shape as one of the added dimensions
            zero_tensor = torch.zeros(features.shape[0], 1, features.shape[2], features.shape[3], features.shape[4])
            
            # Concatenate the zero tensor with the original tensor
            features = torch.cat((zero_tensor.to(features.device), features), dim=1)
            features = features.view(features.shape[0], features.shape[1], -1)
            
        return features


    def get_depth(self, x):

        B, T, C, H, W = x.shape

        x_ = x.flatten(0, 1)

        with torch.no_grad():
            outputs = self.depth_model(x_, output_hidden_states=True)

        feats = outputs.hidden_states[-1][:, :-1]

        feats = feats.view(B, T, -1)
        return feats

    def get_segmentation(self, x):

        B, T, C, H, W = x.shape

        x = x.flatten(0, 1)  # .cuda()

        with torch.no_grad():
            outputs = self.seg_model(pixel_values=x, output_hidden_states=True)

        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        featues = outputs.hidden_states[-1]

        featues = featues.view(B, T, -1)
        return featues

    def forward(self, videos):
        '''
        videos:  [[Tensor[batch x channel x time x height x width] ... n_views] ... n_clips]
        returns: views * [Tensor[B, K, D]] extracted features. The K tokens includes the features across many clips over time
        batch, channel, time, height, width: 16, 3, 16, 224, 224
        '''
        flow_features = self.get_flow(videos)

        depth_features = self.get_depth(videos)

        seg_features = self.get_segmentation(videos).to(torch.float32)

        merged_feat = torch.cat([flow_features, depth_features, seg_features], 2)

        return merged_feat

# Given sequence of images, predicts next latent
class FeatsModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.encoder = FeatsEncoder(model_name)

    def forward(self, x):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        assert len(x.shape) == 5
        return self.encoder(x)
