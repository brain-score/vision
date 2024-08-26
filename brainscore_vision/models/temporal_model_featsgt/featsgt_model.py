from transformers import MaskFormerForInstanceSegmentation
import torch
import torch.nn as nn

from jepa.evals.feature_extract_interface import ActivityRecogFeatureExtractor

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import raft_core.raft as raft # this needs to be added to phys_readouts
from einops import rearrange
import os


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


class FeatsEncoder(ActivityRecogFeatureExtractor):
    def __init__(self, embed_dim=256):
        super().__init__()

        self.mean = torch.tensor([[0.485, 0.456, 0.406]])[:, :, None, None, None]

        self.std = torch.tensor([[0.229, 0.224, 0.225]])[:, :, None, None, None]

        self.embed_dim = embed_dim

        self.pretrained_raft_model = raft.load_raft_model(
            load_path=os.path.expanduser(
                '/ccn2/u/rmvenkat/code/deploy_code/BBNet/bbnet/models/raft_core/raft-sintel.pth'),
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

        #         torch.Size([2, 1568, 896]) torch.Size([2, 512, 3072]) torch.Size([2, 200, 2048])

        n_flow = 1920
        n_depth = 6144
        n_seg = 4096
        pe_len = 2280

        self.fc_flow = nn.Sequential(
            nn.Linear(n_flow, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.fc_depth = nn.Sequential(
            nn.Linear(n_depth, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.fc_seg = nn.Sequential(
            nn.Linear(n_seg, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

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

        return x * self.std.to(x.device) + self.mean.to(x.device)

    def get_flow(self, videos):
        '''
        videos: [B, 3, 16, 224, 224]
        return features of shape: [B, N_patches, D]
        '''

        videos = self.unnormalize(videos)

        videos = videos.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            flow, features = self.pretrained_raft_model(videos)
            features = features.flatten(1, 2).flatten(2, 3).permute(0, 2, 1)
            
        return features


    def get_depth(self, x):

        B, C, T, H, W = x.shape

        x_ = x.transpose(1, 2).flatten(0, 1)

        with torch.no_grad():
            outputs = self.depth_model(x_, output_hidden_states=True)
            predicted_depth = outputs.predicted_depth
            
        predicted_depth = predicted_depth.view(B, T, predicted_depth.shape[1], predicted_depth.shape[2])[:, :, None]

        predicted_depth = patchify(predicted_depth, 1, (8, 8))

        return predicted_depth

    def get_segmentation(self, x):

        B, C, T, H, W = x.shape

        x = x.transpose(1, 2).flatten(0, 1)  # .cuda()

        with torch.no_grad():
            outputs = self.seg_model(pixel_values=x, output_hidden_states=True)

        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        featues = outputs.hidden_states[-1]

        featues = featues.view(B, T, featues.shape[1], featues.shape[2]).permute(0, 2, 1, 3).flatten(2, 3)

        return featues


    def reorder_features(self, outputs, B, num_views_per_clip, num_clips):

        # Unroll outputs into a 2D array [spatial_views x temporal_views]

        eff_B = B * num_views_per_clip
        all_outputs = [[] for _ in range(num_views_per_clip)]
        for i in range(num_clips):
            o = outputs[i * eff_B:(i + 1) * eff_B]
            for j in range(num_views_per_clip):
                all_outputs[j].append(o[j * B:(j + 1) * B])
            # concatenate along first dimension

        for i in range(num_views_per_clip):
            all_outputs[i] = torch.cat(all_outputs[i], dim=1)

        return all_outputs

    def forward(self, videos):
        '''
        videos:  [[Tensor[batch x channel x time x height x width] ... n_views] ... n_clips]
        returns: views * [Tensor[B, K, D]] extracted features. The K tokens includes the features across many clips over time
        batch, channel, time, height, width: 16, 3, 16, 224, 224
        '''

        x = videos

        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, T, H, W = x[0][0].size()

        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)

        # print("flow", self.get_flow(x).shape, "depth", self.get_depth(x).shape, "seg", self.get_segmentation(x).shape)

        flow_features = self.fc_flow(self.get_flow(x))

        depth_features = self.fc_depth(self.get_depth(x))

        seg_features = self.fc_seg(self.get_segmentation(x).to(torch.float32))

        merged_feat = torch.cat([flow_features, depth_features, seg_features], 1)

        merged_feat = self.reorder_features(merged_feat, B, num_views_per_clip, num_clips)

        return merged_feat

# Given sequence of images, predicts next latent
class FeatsModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.encoder = FeatsEncoder()

    def forward(self, x):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        assert len(x.shape) == 5
        return self.encoder(x)
