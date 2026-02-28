from torch import nn

import torch

import phys_extractors.models.jepa_physics.jepa.src.models.vision_transformer as vit


class VJEPA_encoder(nn.Module):
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

        # set requires_grad false
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, videos):
        B, T, C, H, W = videos.shape
        embeddings = self.encoder(videos.transpose(1,2))
        embeddings = self.patchify(embeddings)
        return embeddings

    def patchify(self, input_array, patch_height=32, patch_width=32):
        # Step 2: Pad the matrix to size 392x1036
        padding_size = 12  # Number of columns to add
        padded_matrix = torch.nn.functional.pad(input_array, (0, padding_size))
        # Step 3: Define patch size
        N, _, _ = padded_matrix.shape
        H_prime = 56
        W_prime = 148

        # Step 4: Patchify the padded matrix into 49 contiguous patches of size 56x148
        # Reshape the matrix into shape (7, 56, 7, 148) and then swap axes to obtain patches
        patches = padded_matrix.unfold(1, H_prime, H_prime)
        patches = patches.unfold(2, W_prime, W_prime)
        patches = patches.contiguous().view(N, -1, H_prime, W_prime)  # 49 patches of size 56x148

        return patches

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
