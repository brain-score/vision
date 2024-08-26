from torch import nn

import torch

from jepa.evals.feature_extract_interface import ActivityRecogFeatureExtractor

import jepa.src.models.vision_transformer as vit


class VJEPA_encoder(ActivityRecogFeatureExtractor):
    def __init__(self, weights_path, embed_dim=256):
        super().__init__()


        self.embed_dim = embed_dim

        # download the model and put it in the folder.
        state_dict = torch.load(weights_path)

        # following the config for the model
        crop_size = 224
        patch_size = 16
        num_frames = 16
        tubelet_size = 2

        uniform_power = True
        use_sdpa = True
        use_SiLU = False
        tight_SiLU = False

        self.encoder = vit.__dict__['vit_large'](
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
            use_SiLU=use_SiLU,
            tight_SiLU=tight_SiLU,
        )

        self.encoder.load_state_dict({k.replace('module.backbone.', ''): v for k, v in state_dict['encoder'].items()})
        self.encoder.eval()

        self.fc = nn.Sequential(
            nn.Linear(1024, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 3136, embed_dim))

        # set requires_grad false
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_embed_dim(self):
        return self.embed_dim

    def get_num_heads(self):
        return 16

    def get_pos_embed(self):
        return None

    def apply_pos_embed(self, outputs, pos_embed):
        return outputs

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

        x = [[videos]]

        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, T, H, W = x[0][0].size()

        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)

        embeddings = self.reorder_features(self.fc(self.encoder(x)), B, num_views_per_clip, num_clips)

        return embeddings

# Given sequence of images, predicts next latent
class VJEPA(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.encoder = VJEPA_encoder(model_name)

    def forward(self, x):
        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        assert len(x.shape) == 5
        return self.encoder(x)