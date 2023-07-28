import logging
import numpy as np
from collections import defaultdict
import torch
import torch.nn.functional as F
import imageio
from skimage.draw import line_aa
import mlflow

from physopt.objective.utils import PRETRAINING_PHASE_NAME, READOUT_PHASE_NAME
from physopt.objective import PretrainingObjectiveBase, ExtractionObjectiveBase
from physion.objective.objective import PytorchModel
from physion.data.pydata import TDWDatasetBase

from neuralphys.models.rpin import Net
from neuralphys.utils.misc import tprint

from scipy.spatial.transform import Rotation as R
import os
import io
import glob
import h5py
import json
from PIL import Image
from torchvision import transforms

class RPINModel(PytorchModel):
    def get_model(self):
        model = Net(self.pretraining_cfg.MODEL)
        return model.to(self.device)

def xyxy_to_xywh(x1, y1, x2, y2):
    x = (x1+x2)/2
    y = (y1+y2)/2
    w = (x2-x1)/2
    h = (y2-y1)/2
    return x,y,w,h

def xywh_to_xyxy(x, y, w, h):
    x1 = x - w
    x2 = x + w
    y1 = y - h
    y2 = y + h
    return x1,y1,x2,y2

def compute_bboxes_from_mask(id_img, colors, tol=0.1):
    bboxes = []
    for color in colors:
        if not color.any(): # all zeros
            continue
        idxs = np.argwhere(id_img==color)
        if len(idxs) > 0:
            x1 = np.min(idxs[:,1])
            x2 = np.max(idxs[:,1])
            y1 = np.min(idxs[:,0])
            y2 = np.max(idxs[:,0])
            xyxy = np.array([x1,y1,x2,y2])
            x,y,w,h = xyxy_to_xywh(*xyxy)
            w, h = w*(1+tol), h*(1+tol)
            xyxy = np.clip(xywh_to_xyxy(x,y,w,h), 0, id_img.shape[0]-1).astype(np.uint8)
        else:
            xyxy = -np.ones(4)
        bboxes.append(xyxy)
    return bboxes

def get_colors(f):
    id_img = np.array(f['frames']['0000']['images']['_id'][()]) # first frame, assumes all objects are in view
    id_img = np.array(Image.open(io.BytesIO(id_img))) # (256, 256, 3)
    colors = np.unique(id_img.reshape(-1, id_img.shape[2]), axis=0) # full list of unique colors in id map
    return colors

class TDWDataset(TDWDatasetBase):
    def __getitem__(self, index):
        with h5py.File(self.hdf5_files[index], 'r') as f: # load ith hdf5 file from list
            colors = get_colors(f)
            frames = list(f['frames'])
            target_contacted_zone = False
            for frame in reversed(frames):
                lbl = f['frames'][frame]['labels']['target_contacting_zone'][()]
                if lbl: # as long as one frame touching, label is True
                    target_contacted_zone = True
                    break

            assert len(frames)//self.subsample_factor >= self.seq_len, 'Images must be at least len {}, but are {}'.format(self.seq_len, len(frames)//self.subsample_factor)
            if self.random_seq: # randomly sample sequence of seq_len
                start_idx = self.rng.randint(len(frames)-(self.seq_len*self.subsample_factor)+1)
            else: # get first seq_len # of frames
                start_idx = 0
            end_idx = start_idx + (self.seq_len*self.subsample_factor)
            images = []
            img_transforms = transforms.Compose([
                transforms.Resize((self.imsize, self.imsize)),
                transforms.ToTensor(),
                ])
            rois = []
            # object_ids = np.array(f['static']['object_ids'])
            prev_bboxes = 0
            for frame in frames[start_idx:end_idx:self.subsample_factor]:
                img = f['frames'][frame]['images']['_img'][()]
                if img.ndim == 1:
                    img = Image.open(io.BytesIO(img)) # (256, 256, 3)
                else:
                    img = Image.fromarray(img)
                img = img_transforms(img) # TODO: also need to rescale bboxes if resizing image
                images.append(img)
                if 'bboxes' in f['frames'][frame]:
                    bboxes = f['frames'][frame]['bboxes'][()]
                else:
                    id_img = f['frames'][frame]['images']['_id'][()]
                    id_img = np.array(Image.open(io.BytesIO(id_img))) # (256, 256, 3)
                    bboxes = compute_bboxes_from_mask(id_img, colors)
                # bboxes = []
                # # print(len(object_ids), object_ids)
                # # print([k for k in f['static']['mesh'].keys() if 'vertices' in k])
                # for i, obj_id in enumerate(object_ids):
                #     obj_id = i # TODO
                #     vertices_orig, faces_orig = get_vertices_scaled(f, obj_id)
                #     if len(vertices_orig) == 0 or len(faces_orig) == 0: # TODO
                #         continue
                #     all_pts, all_edges, all_faces = get_full_bbox(vertices_orig)
                #     frame_pts = get_transformed_pts(f, all_pts, frame, obj_id)
                #     bbox = (compute_bboxes(frame_pts, f) * (self.imsize-1)) # scale from [0,1] to [0,H/W]
                #     bboxes.append(bbox)
                # bboxes = np.clip(bboxes, 0, None) # convert -1 for occluded objects to 0
                bboxes, prev_bboxes = np.where(bboxes==-1, prev_bboxes, bboxes), bboxes
                rois.append(bboxes)

            rois = np.array(rois, dtype=np.float32)
            num_objs = rois.shape[1]
            max_objs = 10 # self.pretraining_cfg.MODEL.RPIN.NUM_OBJS # TODO: do padding elsewhere?
            assert num_objs <= max_objs, f'num objs {num_objs} greater than max objs {max_objs}'
            ignore_mask = np.ones(max_objs, dtype=np.float32)
            if num_objs < max_objs:
                rois = np.pad(rois, [(0,0), (0, max_objs-num_objs), (0,0)])
                ignore_mask[num_objs:] = 0
            labels = torch.from_numpy(build_labels(rois)[self.state_len-1:])
            rois = torch.from_numpy(rois)
            images = torch.stack(images, dim=0)
            binary_labels = torch.ones((self.seq_len, 1)) if target_contacted_zone else torch.zeros((self.seq_len, 1)) # Get single label over whole sequence
            stimulus_name = f['static']['stimulus_name'][()]

        sample = {
            'data': images[:self.state_len],
            'rois': rois,
            'labels': labels, # [off, pos]
            'data_last': images[:self.state_len],
            'ignore_mask': torch.from_numpy(ignore_mask),
            'stimulus_name': stimulus_name,
            'binary_labels': binary_labels,
            'images': images,
        }
        return sample

def build_labels(rois):
        assert rois.ndim == 3, rois.shape # (T, K, 4)
        assert rois.shape[-1] == 4, rois.shape
        pos = (rois[:,:,:2] + rois[:,:,2:] ) / 2 # get x,y pos
        off = pos[1:] - pos[:-1] # get x,y offset
        labels = np.concatenate([off, pos[1:]], axis=-1)
        return labels

def save_vis(frames, rois, labels, bbox, ignore_idx, stimulus_name, output_dir, prefix=0, artifact_path='videos', n_vis=1):
    # print(frames.shape, rois.shape, labels.shape, bbox.shape)
    BS, T, _, H, W = frames.shape
    fps = 30 / 9
    for i in range(min(n_vis, BS)):
        curr_stim = stimulus_name[i]
        if type(curr_stim) == bytes:
            curr_stim = curr_stim.decode('utf-8')
        # labels = build_labels(rois[i].numpy()).astype(np.uint8)
        arr = []
        images = torch.permute(255*frames[i], (0,2,3,1)).numpy().astype(np.uint8)
        n_objs = int(ignore_idx[i].sum().item()) # ignore_idx: (BS, K)
        for t in range(T):
            image = images[t]
            for k in range(n_objs):
                off = T-labels.shape[1]
                if t >= off:
                    x,y = labels[i,t-off,k,-2:].numpy().astype(np.uint8)
                    image[y-1:y+1,x-1:x+1] = np.ones((1,3)) * 255
                    x, y = bbox[i, t-off,k,-2:].numpy().astype(np.uint8)
                    image[y-2:y+2,x-2:x+2] = np.array([[255,0,0]])
                image = add_bbox(image, rois[i,t,k])
            arr.append(image)
        arr = np.stack(arr)

        fn = os.path.join(output_dir, f'{prefix:06}_{i:02}_{curr_stim}.mp4')
        imageio.mimwrite(fn, arr, fps=fps)
        mlflow.log_artifact(fn, artifact_path=artifact_path)

def add_bbox(image, roi, color=np.array([255,0,0])):
    x1, y1, x2, y2 = roi.numpy().astype(np.uint8)
    if x1==x2 and y1==y2:
        return image
    image = add_line(image, y1, x1, y1, x2)
    image = add_line(image, y2, x1, y2, x2)
    image = add_line(image, y1, x1, y2, x1)
    image = add_line(image, y1, x2, y2, x2)
    return image

def add_line(image, y1, x1, y2, x2):
    # rr, cc, val = weighted_line(y1, x1, y2, x2, 5)
    rr, cc, val = line_aa(y1, x1, y2, x2)
    image[rr, cc] = val.reshape(-1,1).astype(np.uint8) * 255
    return image

class ExtractionObjective(RPINModel, ExtractionObjectiveBase):
    def get_readout_dataloader(self, datapaths):
        random_seq = False # get sequence from beginning for feature extraction
        shuffle = False # no need to shuffle for feature extraction
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle, 0)

    def extract_feat_step(self, data):
        self.model.eval()
        with torch.no_grad():
            labels = data['binary_labels'].cpu().numpy()
            stimulus_name = np.array(data['stimulus_name'], dtype=object)
            images, boxes, data_last, ignore_idx = [data[k] for k in ['images', 'rois', 'data_last', 'ignore_mask']]
            images = images.to(self.device)
            rois, coor_features = init_rois(boxes, images.shape)
            rois = rois.to(self.device)
            coor_features = coor_features.to(self.device)
            ignore_idx = ignore_idx.to(self.device)
            outputs = self.model(images, rois, coor_features, num_rollouts=self.pretraining_cfg.MODEL.RPIN.PRED_SIZE_TEST,
                                 data_pred=data_last, phase='test', ignore_idx=ignore_idx)
        input_states = torch.flatten(outputs['input_states'], 2).cpu().numpy()
        observed_states = torch.flatten(outputs['encoded_states'], 2).cpu().numpy()
        simulated_states = torch.flatten(outputs['rollout_states'], 2).cpu().numpy()

        save_vis(data['images'], data['rois'], data['labels'], outputs['bbox'].cpu(), ignore_idx, stimulus_name, self.output_dir, self.step, f'videos/{self.mode}')
            
        output = {
            'input_states': input_states,
            'observed_states': observed_states,
            'simulated_states': simulated_states,
            'labels': labels,
            'stimulus_name': stimulus_name,
            }
        # print([(k,v.shape) for k,v in output.items()])
        return output

class PretrainingObjective(RPINModel, PretrainingObjectiveBase):
    def get_pretraining_dataloader(self, datapaths, train):
        random_seq = True # get random slice of video during pretraining
        shuffle = True if train else False # no need to shuffle for validation
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle, 0)
        # return self.get_dataloader(TDWDataset, datapaths, False, False, 0)

    def train_step(self, data):
        self.model.train() # set to train mode
        self._init_loss()
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.pretraining_cfg.TRAIN.LR,
            weight_decay=self.pretraining_cfg.TRAIN.WEIGHT_DECAY,
        )
        
        # self._adjust_learning_rate() # TODO
        stimulus_name = np.array(data['stimulus_name'], dtype=object)
        images, boxes, labels, data_last, ignore_idx = [data[k] for k in ['data', 'rois', 'labels', 'data_last', 'ignore_mask']]

        images = images.to(self.device)
        labels = labels.to(self.device)
        rois, coor_features = init_rois(boxes, images.shape)
        rois = rois.to(self.device)
        coor_features = coor_features.to(self.device)
        ignore_idx = ignore_idx.to(self.device)
        # print(images.dtype, labels.dtype, rois.dtype, coor_features.dtype, ignore_idx.dtype)
        optim.zero_grad()
        outputs = self.model(images, rois, coor_features, num_rollouts=self.ptrain_size,
                             data_pred=data_last, phase='train', ignore_idx=ignore_idx)
        loss = self.loss(outputs, labels, 'train', ignore_idx)
        loss.backward()
        optim.step()

        vis_freq = getattr(self.pretraining_cfg.TRAIN, 'VIS_FREQ', 100*self.pretraining_cfg.LOG_FREQ) # use 100*log_freq as vis_freq if not found 
        if self.step % vis_freq == 0:
            self.model.eval()
            save_vis(data['images'], data['rois'], data['labels'], outputs['bbox'].detach().cpu(), ignore_idx, stimulus_name, self.output_dir, self.step, f'videos/train')

        return loss.item() # scalar loss value for the step

    def val_step(self, data):
        with torch.no_grad():
            self.model.eval() # set to eval mode
            self._init_loss()
            data, boxes, labels, data_last, ignore_idx = [data[k] for k in ['data', 'rois', 'labels', 'data_last', 'ignore_mask']]

            data = data.to(self.device)
            labels = labels.to(self.device)
            rois, coor_features = init_rois(boxes, data.shape)
            rois = rois.to(self.device)
            coor_features = coor_features.to(self.device)
            ignore_idx = ignore_idx.to(self.device)
            outputs = self.model(data, rois, coor_features, num_rollouts=self.ptest_size,
                                 data_pred=data_last, phase='test', ignore_idx=ignore_idx)
            loss = self.loss(outputs, labels, 'test', ignore_idx)

        val_res = {
            'val_loss': loss.item(),
        }
        return val_res

    def _init_loss(self):
        cfg = self.pretraining_cfg.MODEL.RPIN
        self.loss_name = []
        self.offset_loss_weight = cfg.OFFSET_LOSS_WEIGHT
        self.position_loss_weight = cfg.POSITION_LOSS_WEIGHT
        self.loss_name += ['p_1', 'p_2', 'o_1', 'o_2']
        if cfg.VAE:
            self.loss_name += ['k_l']
        self.ptrain_size = cfg.PRED_SIZE_TRAIN
        self.ptest_size = cfg.PRED_SIZE_TEST
        self.losses = dict.fromkeys(self.loss_name, 0.0)
        self.pos_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.off_step_losses = [0.0 for _ in range(self.ptest_size)]
        # an statistics of each validation

    def loss(self, outputs, labels, phase='train', ignore_idx=None): # TODO: just pass bbox_rollouts instead of full output?
        C = self.pretraining_cfg.MODEL
        valid_length = self.ptrain_size if phase == 'train' else self.ptest_size

        bbox_rollouts = outputs['bbox'] # of shape (batch, time, #obj, 4)
        rollout_steps = bbox_rollouts.shape[1]
        # print(bbox_rollouts[0,:,:3], labels[0,:rollout_steps,:3])
        loss = (bbox_rollouts - labels[:,:rollout_steps]) ** 2 # (BS, rollout_len, 4)
        # loss /= (self.pretraining_cfg.DATA.IMSIZE - 1)**2 # normalize by imsize, squared since loss is squared
        # take mean except time axis, time axis is used for diagnosis
        ignore_idx = ignore_idx[:, None, :, None].to('cuda')
        loss = loss * ignore_idx
        loss = loss.sum(2) / ignore_idx.sum(2)
        loss[..., 0:2] = loss[..., 0:2] * self.offset_loss_weight
        loss[..., 2:4] = loss[..., 2:4] * self.position_loss_weight
        # o_loss = loss[..., 0:2]  # offset
        # p_loss = loss[..., 2:4]  # position

        # for i in range(valid_length):
        #     self.pos_step_losses[i] += p_loss[:, i].sum(0).sum(-1).mean().item()
        #     self.off_step_losses[i] += o_loss[:, i].sum(0).sum(-1).mean().item()

        # p1_loss = self.pos_step_losses[:self.ptrain_size]
        # p2_loss = self.pos_step_losses[self.ptrain_size:]
        # self.losses['p_1'] = np.mean(p1_loss)
        # self.losses['p_2'] = np.mean(p2_loss)

        # o1_loss = self.off_step_losses[:self.ptrain_size]
        # o2_loss = self.off_step_losses[self.ptrain_size:]
        # self.losses['o_1'] = np.mean(o1_loss)
        # self.losses['o_2'] = np.mean(o2_loss)

        # no need to do precise batch statistics, just do mean for backward gradient
        # loss = loss.mean(0)
        # pred_length = loss.shape[0]
        # init_tau = C.RPIN.DISCOUNT_TAU ** (1 / self.ptrain_size)
        # tau = init_tau + (self.step / self.pretraining_cfg.TRAIN_STEPS) * (1 - init_tau)
        # tau = torch.pow(tau, torch.arange(pred_length, out=torch.FloatTensor()))[:, None]
        # # tau = torch.cat([torch.ones(self.cons_size, 1), tau], dim=0).to('cuda')
        # tau = tau.to(self.device)
        # loss = ((loss * tau) / tau.sum(axis=0, keepdims=True)).sum()
        loss = loss.mean(0).sum()

        if C.RPIN.VAE and phase == 'train':
            kl_loss = outputs['kl_loss']
            self.losses['k_l'] += kl_loss.sum().item()
            loss += C.RPIN.VAE_KL_LOSS_WEIGHT * kl_loss.sum()

        return loss

def init_rois(boxes, shape):
    batch, time_step, _, height, width = shape
    max_objs = boxes.shape[2]
    # coor features, normalized to [0, 1]
    num_im = batch * time_step
    # noinspection PyArgumentList
    co_f = np.zeros(boxes.shape[:-1] + (2,))
    co_f[..., 0] = torch.mean(boxes[..., [0, 2]], dim=-1).numpy().copy() / width
    co_f[..., 1] = torch.mean(boxes[..., [1, 3]], dim=-1).numpy().copy() / height
    coor_features = torch.from_numpy(co_f.astype(np.float32))
    rois = boxes[:, :time_step]
    batch_rois = np.zeros((num_im, max_objs))
    batch_rois[np.arange(num_im), :] = np.arange(num_im).reshape(num_im, 1)
    # noinspection PyArgumentList
    batch_rois = torch.FloatTensor(batch_rois.reshape((batch, time_step, -1, 1)))
    rois = torch.cat([batch_rois, rois], dim=-1)
    return rois, coor_features

def get_camera_matrix(f):
    projection_matrix =  np.array(f['frames']['0000']['camera_matrices']['projection_matrix']).reshape(4,4)
    camera_matrix =  np.array(f['frames']['0000']['camera_matrices']['camera_matrix']).reshape(4,4)
    return np.matmul(projection_matrix, camera_matrix)

def project_points(points, camera_matrix):
    assert points.ndim == 2
    assert points.shape[1] == 3
    
    points = np.pad(points, [(0,0), (0,1)], constant_values=1).T
    projected_points = np.matmul(camera_matrix, points).T
    projected_points = projected_points / projected_points[:,-1:]
    return projected_points

def compute_bbox_from_projected_pts(points):
    assert points.ndim == 2
    assert points.shape[1] == 4
    x1 = np.min(points[:,0])
    y1 = np.min(-points[:,1]) # flip y
    x2 = np.max(points[:,0])
    y2 = np.max(-points[:,1]) # flip y
    xyxy = rescale_xyxy(np.array([x1, y1, x2, y2]))
    return xyxy

def compute_bboxes(points, f):
    camera_matrix = get_camera_matrix(f)
    ppoints = project_points(points, camera_matrix)
    xyxy = compute_bbox_from_projected_pts(ppoints)
    return xyxy

def rescale_xyxy(xyxy):
    xyxy = np.clip(xyxy, -1, 1) # ensure [-1,1]
    xyxy = (xyxy + 1.) / 2. # scale to [0,1]
    return xyxy

def get_vertices_scaled(f, obj_id):
    
    vertices_orig = np.array(f['static']['mesh']['vertices_' + str(obj_id)])

    scales = f["static"]["scale"][:]

    vertices_orig[:,0] *= scales[obj_id, 0]
    vertices_orig[:,1] *= scales[obj_id, 1]
    vertices_orig[:,2] *= scales[obj_id, 2]
    faces_orig = np.array(f['static']['mesh']['faces_' + str(obj_id)])
    
    return vertices_orig, faces_orig

def get_full_bbox(vertices):
    arr1 = vertices.min(0)
    
    arr2 = vertices.max(0)
    
    arr = np.stack([arr1, arr2], 0)
    
    pts = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0) , (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]
         
    all_edges = [(0, 1), (1, 2), (2, 3), (3, 0),  (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    
    all_faces = [(0, 1, 2, 3), (4, 5, 6, 7), (2, 3, 6, 7), (1, 0, 4, 5), (1, 2, 6, 5), \
                (0, 4, 7, 3)]    
    
    index = np.arange(3)
    
    all_pts = []
    for pt in pts:
        p1 = arr[pt, index]
        all_pts.append(p1)
    
    all_pts = np.stack(all_pts, 0)
    
    return all_pts, all_edges, all_faces    


def get_transformed_pts(f, pts, frame, obj_id):
    rotations_0 = np.array(f['frames'][frame]['objects']['rotations'][obj_id])
    positions_0 = np.array(f['frames'][frame]['objects']['positions'][obj_id])
    
    rot = R.from_quat(rotations_0).as_matrix()
    trans = positions_0
    transformed_pts = np.matmul(rot, pts.T).T + np.expand_dims(trans, axis=0)
    
    return transformed_pts

