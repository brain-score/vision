import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from phys_extractors.models.mcvd_pytorch.load_model_from_ckpt import load_model, get_readout_sampler, init_samples
from phys_extractors.models.mcvd_pytorch.datasets import data_transform
from phys_extractors.models.mcvd_pytorch.runners.ncsn_runner import conditioning_fn
    
class MCVD(nn.Module):
    def __init__(self, weights_path, identifier):
        
        super().__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('device: ', device)
        if torch.cuda.is_available():
            print("CUDA is available!")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("CUDA is not available.")

        if 'EGO4D' in identifier:
            cfg_path = 'config_ego4d.yml'
        else:
            cfg_path = 'config_physion.yml'
        self.scorenet, self.config = load_model(weights_path, device, cfg_path)
        self.sampler = get_readout_sampler(self.config)

    def forward(self, videos):
        #videos = torch.stack([self.transform_video_tensor(vid) for vid in videos])
        input_frames = data_transform(self.config, videos)
        if self.config.data.num_frames_cond+self.config.data.num_frames > videos.shape[1]:
            added_frames = self.config.data.num_frames_cond+self.config.data.num_frames - videos.shape[1]
            input_frames = torch.cat([input_frames] + [input_frames[:, -1].unsqueeze(1)]*added_frames, axis=1)
        output = []
        for j in range(0, videos.shape[1], self.config.data.num_frames_cond+self.config.data.num_frames):
            real, cond, cond_mask = conditioning_fn(self.config, 
                                                    input_frames[:, 
                                                    j:j+self.config.data.num_frames_cond+self.config.data.num_frames, 
                                                    :, :, :], 
                                        num_frames_pred=self.config.data.num_frames,
                                        prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                        prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0))

            init = init_samples(len(real), self.config)
            with torch.no_grad():
                pred, gamma, beta, mid = self.sampler(init, self.scorenet, cond=cond,
                                         cond_mask=cond_mask,
                                         subsample=100, verbose=True)
            output +=  [mid]
        return torch.stack(output, axis=1)

# Given sequence of images, predicts next latent
class MCVDEncoder(nn.Module):
    def __init__(self, weights_path, identifier):
        super().__init__()
        self.encoder = MCVD(weights_path, identifier)

    def forward(self, x):
        self.encoder.eval()
        output = self.encoder(x)
        return output
