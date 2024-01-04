import os
import numpy as np
import logging
import imageio
import torch
import torch.nn as nn
import mlflow
import scipy
import cv2
from skimage.metrics import structural_similarity
from lpips import LPIPS
from  physion.frechet_video_distance import fvd as tf_fvd

from physopt.objective import PretrainingObjectiveBase, ExtractionObjectiveBase
from physion.objective.objective import PytorchModel
from physion.data.pydata import TDWDataset
from physion.models.fitvid import FitVid

N_VIS_PER_BATCH = 1
BASE_FPS = 30

class CustomDataParallel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = nn.DataParallel(model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

class FitVidModel(PytorchModel):
    def get_model(self):
        model = FitVid(
            input_size=3, 
            n_past=self.pretraining_cfg.DATA.STATE_LEN, 
            **self.pretraining_cfg.MODEL
            )
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logging.info(f'Using {torch.cuda.device_count()} gpus')
            model = CustomDataParallel(model)
        return model.to(self.device)

    def save_model(self, model_file):
        logging.info(f'Saved model checkpoint to: {model_file}')
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            torch.save(self.model.model.module.state_dict(), model_file) # TODO: cleanup
        else:
            torch.save(self.model.state_dict(), model_file)

    def load_model(self, model_file):
        assert os.path.isfile(model_file), f'Cannot find model file: {model_file}'
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model.model.module.load_state_dict(torch.load(model_file)) # TODO: cleanup
        else:
            self.model.load_state_dict(torch.load(model_file))
        logging.info(f'Loaded existing ckpt from {model_file}')
        return self.model

class PretrainingObjective(FitVidModel, PretrainingObjectiveBase):
    def get_pretraining_dataloader(self, datapaths, train):
        random_seq = True # get random slice of video during pretraining
        shuffle = True if train else False # no need to shuffle for validation
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle, num_workers=0)

    def setup(self):
        super().setup()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.pretraining_cfg.TRAIN.LR)
        self.optimizer.zero_grad()
        if 'ACCUMULATION_BATCH_SIZE' in self.pretraining_cfg.TRAIN:
            assert self.pretraining_cfg.TRAIN.ACCUMULATION_BATCH_SIZE % self.pretraining_cfg.BATCH_SIZE == 0, \
                f'accumulation batch size ({self.pretraining_cfg.TRAIN.ACCUMULATION_BATCH_SIZE}) not divisible by batch size ({self.pretraining_cfg.BATCH_SIZE})'
            self.accumulation_steps = self.pretraining_cfg.TRAIN.ACCUMULATION_BATCH_SIZE // self.pretraining_cfg.BATCH_SIZE
        else: # for backwards compatibility
            self.accumulation_steps = self.pretraining_cfg.TRAIN.ACCUMULATION_STEPS
        logging.info(f'Using {self.accumulation_steps} accumulation steps of size {self.pretraining_cfg.BATCH_SIZE}')

    def train_step(self, data):
        self.model.training = True
        self.model.train()

        model_output = self.model(data['images'].to(self.device)) # train video length = 12
        loss = model_output['loss'].mean() # assumes batch size for each gpu is the same
        loss_val = loss.item() # get loss val before normalizing
        loss = loss / self.accumulation_steps # normalize loss since using average
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e2)
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        vis_freq = getattr(self.pretraining_cfg.TRAIN, 'VIS_FREQ', 100*self.pretraining_cfg.LOG_FREQ) # use 100*log_freq as vis_freq if not found 
        if self.step % vis_freq == 0:
            # get model preds under eval mode
            self.model.training = False
            self.model.eval()
            model_output_eval = self.model(data['images'].to(self.device))
            # save visualizations
            frames = {
                'gt': data['images'],
                'sim': model_output_eval['preds'].cpu().detach(),
                'stimulus_name': data['stimulus_name'],
                }
            save_vis(frames, self.pretraining_cfg, self.output_dir, self.step, 'videos/train')
        return loss_val

    def val_step(self, data):
        self.model.training = False
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(data['images'].to(self.device))
            loss = model_output['loss'].mean() # assumues batch size for each gpu is the same
            # only compare part after input frames
            out_video = model_output['preds'][:, self.model.n_past:].cpu().numpy()
            gt = data['images'][:, self.model.n_past:].numpy()

        val_res =  {'val_loss': loss.item()}
        val_res['psnr'] = psnr(gt, out_video, max_val=1.)
        val_res['ssim'] = ssim(gt, out_video, max_val=1.)
        val_res['lpips'] = lpips(gt, out_video)
        # val_res['fvd'] = fvd(gt, out_video)

        # save visualizations
        frames = {
            'gt': data['images'],
            'sim': model_output['preds'].cpu(),
            'stimulus_name': data['stimulus_name'],
            }
        save_vis(frames, self.pretraining_cfg, self.output_dir, self.vstep, f'videos/val/{self.step}')

        return val_res

def preprocess_video(video, permute=True, merge=True):
    if permute:
        video =  np.transpose(video, (0,1,3,4,2)) # put channels last
    if merge:
        video = np.reshape(video, (-1,) + video.shape[2:]).astype(np.float32)
    return video

def psnr(video_1, video_2, max_val):
    video_1 = preprocess_video(video_1)
    video_2 = preprocess_video(video_2)
    assert video_1.shape == video_2.shape, (video_1.shape, video_2.shape)
    mse = np.mean(np.square(video_1 - video_2), axis=(-3,-2,-1))
    psnr_val = np.subtract(
            20 * np.log(max_val) / np.log(10.0),
            np.float32(10 / np.log(10)) * np.log(mse))
    return np.mean(psnr_val).tolist()

def ssim(video_1, video_2, max_val):
    video_1 = preprocess_video(video_1)
    video_2 = preprocess_video(video_2)
    assert video_1.shape == video_2.shape
    dist = np.array([structural_similarity(video_1[i], video_2[i], data_range=max_val, channel_axis=2) for i in range(len(video_1))])
    return np.mean(dist).tolist()

def lpips(video_1, video_2):
    with torch.no_grad():
        video_1 = 2 * torch.from_numpy(preprocess_video(video_1, permute=False)) - 1 # normalize [-1,1]
        video_2 = 2 * torch.from_numpy(preprocess_video(video_2, permute=False)) - 1 # normalize [-1,1]
        assert video_1.shape == video_2.shape
        loss_fn_alex = LPIPS(net='alex') # best forward scores
        dist = loss_fn_alex(video_1, video_2)
        return np.mean(dist.numpy()).tolist()

def fvd(video_1, video_2):
    video_1 = preprocess_video(video_1, merge=False)
    video_2 = preprocess_video(video_2, merge=False)
    return tf_fvd(video_1, video_2)

class ExtractionObjective(FitVidModel, ExtractionObjectiveBase):
    def get_readout_dataloader(self, datapaths):
        random_seq = False # get sequence from beginning for feature extraction
        shuffle = False # no need to shuffle for feature extraction
        return self.get_dataloader(TDWDataset, datapaths, random_seq, shuffle, num_workers=0)

    def setup(self):
        super().setup()

    def extract_feat_step(self, data):
        self.model.training = False
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(data['images'].to(self.device))
            preds = model_output['preds']
            h_preds = model_output['h_preds'].cpu().numpy()
            # only compare part after input frames
            out_video = model_output['preds'][:, self.model.n_past:].cpu().numpy()
            gt = data['images'][:, self.model.n_past:].numpy()

            # get observed states
            hidden, skips = self.model.encoder(data['images'].to(self.device))
            observed_hs = torch.sigmoid(hidden)
            observed_preds = self.model.decoder(observed_hs, skips)
            observed_hs = observed_hs.cpu().numpy()

        # val_res = {}
        # val_res['psnr'] = psnr(gt, out_video, max_val=1.)
        # val_res['ssim'] = ssim(gt, out_video, max_val=1.)
        # val_res['lpips'] = lpips(gt, out_video)
        # val_res['fvd'] = fvd(gt, out_video)
        # mlflow.log_metrics(val_res, step=self.step) 

        # save visualizations
        frames = {
            'gt': data['images'],
            'obs': observed_preds.cpu(),
            'sim': preds.cpu(),
            'stimulus_name': data['stimulus_name'],
            }
        save_vis(frames, self.pretraining_cfg, self.output_dir, self.step, f'videos/{self.mode}')

        labels = data['binary_labels'].cpu().numpy()
        stimulus_name = np.array(data['stimulus_name'], dtype=object)
        rollout_len = self.pretraining_cfg.DATA.SEQ_LEN - self.pretraining_cfg.DATA.STATE_LEN
        output = {
            'input_states': observed_hs[:,:-rollout_len], # encoded input frames
            'observed_states': observed_hs[:,-rollout_len:], # encoded future frames
            'simulated_states': h_preds[:,-rollout_len:], # rollout predicted future states
            'labels': labels,
            'stimulus_name': stimulus_name,
            }
        return output
            
def add_border(arr, color=[0,255,0], width_frac=0.01):
    assert type(arr) == np.ndarray
    assert arr.ndim == 4  # (T, H, W, C)
    assert arr.shape[3] == 3, arr.shape

    width = max(int(np.ceil(width_frac * max(*arr.shape[2:]))), 1) # at least 1 pixel wide
    pad_width = [(0,0), (width, width), (width, width)]
    arr = arr[:,width:-width,width:-width,:] # erode image by width, so size is same after adding border
    assert len(color) == 3
    r_cons, g_cons, b_cons = color
    r_, g_, b_ = arr[:, :, :, 0], arr[:, :, :, 1], arr[:, :, :, 2]
    rb = np.pad(array=r_, pad_width=pad_width, mode='constant', constant_values=r_cons)
    gb = np.pad(array=g_, pad_width=pad_width, mode='constant', constant_values=g_cons)
    bb = np.pad(array=b_, pad_width=pad_width, mode='constant', constant_values=b_cons)
    arr = np.stack([rb, gb, bb], axis=-1)
    return arr

def add_rollout_border(arr, rollout_len, target_size=128):
    assert type(arr) == np.ndarray
    assert arr.ndim == 4  # (T, H, W, C)
    assert arr.shape[3] == 3, arr.shape

    H, W = arr.shape[1:3]
    resize_factor = target_size / min(H,W) # resize so min dim is target size
    if resize_factor != 1:
        arr = scipy.ndimage.zoom(arr, (1,resize_factor, resize_factor, 1))
    arr_inp = arr[:-rollout_len]
    arr_inp = add_border(arr_inp, [255, 0, 0])
    arr_pred = arr[-rollout_len:]
    arr_pred = add_border(arr_pred)
    arr = np.concatenate([arr_inp, arr_pred], axis=0)
    return arr

def save_vis(input_frames, pretraining_cfg, output_dir, prefix=0, artifact_path='videos', target_size=128):
    rollout_len = pretraining_cfg.DATA.SEQ_LEN - pretraining_cfg.DATA.STATE_LEN
    fps = BASE_FPS // pretraining_cfg.DATA.SUBSAMPLE_FACTOR
    n_vis_per_batch = min(pretraining_cfg.BATCH_SIZE, N_VIS_PER_BATCH)
    stimulus_name = input_frames.pop('stimulus_name')
    for i in range(n_vis_per_batch):
        arrs = []
        for lbl, arr in input_frames.items():
            curr_stim = stimulus_name[i]
            if type(curr_stim) == bytes:
                curr_stim = curr_stim.decode('utf-8')
            arr = (255*torch.permute(arr[i], (0,2,3,1)).numpy()).astype(np.uint8) # (T,C,H,W) => (T,H,W,C), then [0.,1.] => [0, 255]
            arr = add_rollout_border(arr, rollout_len, target_size)

            # add lbl text to video
            frames = []
            H, W = arr.shape[1:3]
            for frame in arr:
                frame = np.copy(frame) # since text is added to frame in-place
                buf = 0.05 # buffer percentage 
                thickness = 1
                # add black border
                cv2.putText(frame, 
                    text=lbl.upper(),
                    org=(int(buf*W),int((1-buf)*H)), # Point uses (col, row), put in bottom left
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1*min(H,W)/256,
                    color=(0,0,0), # black
                    thickness=2*thickness,
                    lineType=cv2.LINE_AA,
                    )
                # put white text
                cv2.putText(frame, 
                    text=lbl.upper(),
                    org=(int(buf*W),int((1-buf)*H)), # Point uses (col, row), put in bottom left
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1*min(H,W)/256,
                    color=(255,255,255), # white
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                    )
                frames.append(frame)
            arr = np.stack(frames)

            arrs.append(arr)
        arr = np.concatenate(arrs, axis=2) # concatenate along width
        fn = os.path.join(output_dir, f'{prefix:06}_{i:02}_{curr_stim}.mp4')
        imageio.mimwrite(fn, arr, fps=fps, macro_block_size=None)
        logging.info(f'Video written to {fn}')
        if artifact_path is not None:
            mlflow.log_artifact(fn, artifact_path=artifact_path)
