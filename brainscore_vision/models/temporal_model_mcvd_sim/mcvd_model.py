import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from phys_extractors.models.mcvd_pytorch.load_model_from_ckpt import load_model, get_readout_sampler, init_samples
from phys_extractors.models.mcvd_pytorch.datasets import data_transform
from phys_extractors.models.mcvd_pytorch.runners.ncsn_runner import conditioning_fn
    
class MCVD(nn.Module):
    def __init__(self, weights_path, cfg_path, sim_length):
        
        super().__init__()
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.scorenet, self.config = load_model(weights_path, device, cfg_path)
        self.sampler = get_readout_sampler(self.config)
        self.sim_length = sim_length

    def forward(self, videos):
        #videos = torch.stack([self.transform_video_tensor(vid) for vid in videos])
        input_frames = data_transform(self.config, videos)
        if self.config.data.num_frames_cond+self.config.data.num_frames > videos.shape[1]:
            added_frames = self.config.data.num_frames_cond+self.config.data.num_frames - videos.shape[1]
            input_frames = torch.cat([input_frames] + [input_frames[:, -1].unsqueeze(1)]*added_frames, axis=1)

        output = []

        real, cond, cond_mask = conditioning_fn(self.config, 
                                                input_frames[:, 
                                                :self.config.data.num_frames_cond+self.config.data.num_frames, 
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
                
        for j in range(1, self.sim_length):
            if j+self.config.data.num_frames_cond+self.config.data.num_frames <= input_frames.shape[1]:
                real, cond_, cond_mask = conditioning_fn(self.config, 
                                                    input_frames[:, 
                                                    j:j+self.config.data.num_frames_cond+self.config.data.num_frames, 
                                                    :, :, :], 
                                        num_frames_pred=self.config.data.num_frames,
                                        prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                        prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0))
            if self.device == torch.device('cuda'):
                cond = torch.cat((cond[:, 3:, :, :], pred.cuda()), dim=1)
            else:
                cond = torch.cat((cond[:, 3:, :, :], pred), dim=1)
                
            init = init_samples(len(real), self.config)
            with torch.no_grad():
                pred, gamma, beta, mid = self.sampler(init, self.scorenet, cond=cond,
                                         cond_mask=cond_mask,
                                         subsample=100, verbose=True)
            
            output +=  [mid]
        return torch.stack(output, axis=1)

# Given sequence of images, predicts next latent
class MCVDSimulator(nn.Module):
    def __init__(self, weights_path, config_path, sim_length=20):
        super().__init__()
        self.dynamics = MCVD(weights_path, config_path, sim_length)

    def forward(self, x):
        self.dynamics.eval()
        output = self.dynamics(x)
        return output
