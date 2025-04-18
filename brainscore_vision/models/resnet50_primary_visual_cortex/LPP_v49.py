import torch
import torch.nn as nn
from torchvision import transforms


class LPP49(nn.Module):
    def __init__(self, input_h, input_w, output_h, output_w, radius_bins, angle_bins, interpolation=None, subbatch_size=128):
        super(LPP49, self).__init__()

        # Polar coordinate rho and theta vals for each input location
        center_h, center_w = int(input_h/2), int(input_w/2)
        x_coords = torch.arange(input_w).repeat(input_h,1) - center_w
        y_coords = center_h - torch.arange(input_h).unsqueeze(-1).repeat(1,input_w)
        distances = torch.sqrt(x_coords**2 + y_coords**2)
        angles = torch.atan2(y_coords, x_coords)
        angles[y_coords < 0] += 2*torch.pi
        self.distances = distances
        self.angles = angles
        self.radius_bins = radius_bins
        self.angle_bins = angle_bins
        self.n_radii = len(radius_bins)-1
        self.n_angles = len(angle_bins)-1
        self.edge_radius = min(center_h, center_w)

        pooling_masks = []
        for i, (min_dist, max_dist) in enumerate(zip(radius_bins, radius_bins[1:])):
            in_distance = torch.logical_and(distances >= min_dist, distances < max_dist)
            for j, (min_angle, max_angle) in enumerate(zip(angle_bins, angle_bins[1:])):
                in_angle = torch.logical_and(angles >= min_angle, angles < max_angle)
                ind_mask = torch.logical_and(in_distance, in_angle).to(torch.float32)
                pooling_masks.append(ind_mask)
        pooling_masks = torch.stack(pooling_masks)

        if interpolation:
            for mask_idx in range(0, pooling_masks.shape[0], self.n_angles):
                radius = radius_bins[mask_idx//self.n_angles]
                if radius > self.edge_radius:
                    continue
                radius_masks = pooling_masks[mask_idx:mask_idx+self.n_angles]
                nonzero_masks = radius_masks[torch.sum(radius_masks, dim=(1,2)).to(torch.bool)]
                interpolated_masks = torch.nn.functional.interpolate(nonzero_masks.permute(1,2,0), size=self.n_angles, mode=interpolation).permute(2,0,1)
                pooling_masks[mask_idx:mask_idx+self.n_angles] = interpolated_masks
        pooling_mask_counts = torch.sum(pooling_masks, dim=(1,2))
        pooling_mask_counts[pooling_mask_counts == 0] = 1
        self.register_buffer('pooling_masks', pooling_masks)
        self.register_buffer('pooling_mask_counts', pooling_mask_counts)
        self.pooling_masks = self.pooling_masks.half()
        self.pooling_mask_counts = self.pooling_mask_counts.half()
        self.interpolation = interpolation
        self.output_transform = transforms.Resize((output_h, output_w), \
                                    interpolation=transforms.InterpolationMode.BILINEAR)
        self.subbatch_size = subbatch_size


    def forward(self, x):
        n, c, h, w = x.size()
        out = []
        for i in range(0, n, self.subbatch_size):
            # Process batch in subbatches to avoid to reduce memory consumption.  It ain't pretty, but will run on consumer gpu.
            out_i = (torch.sum(torch.mul(x[i:i+self.subbatch_size].unsqueeze(2), self.pooling_masks), dim=(-1,-2)) / self.pooling_mask_counts)
            out_i = out_i.view(out_i.size(0),c,self.n_radii,self.n_angles)
            out.append(out_i)
        out = torch.cat(out)
        out = torch.nn.functional.pad(out, (0,0,1,1), mode='reflect')
        out = torch.nn.functional.pad(out, (1,1,0,0), mode='reflect')
        #out = torch.nn.functional.pad(out, (0,0,1,1))
        #out = torch.nn.functional.pad(out, (1,1,0,0), mode='circular')
        out = self.output_transform(out)
        return out