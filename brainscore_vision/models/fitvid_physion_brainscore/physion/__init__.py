import os
import io
from os.path import expanduser
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import h5py
import numpy as np
import scipy.ndimage
import cv2
import imageio
from PIL import Image
import torch
from torchvision import transforms
from physion.models.fitvid import FitVid

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

PRETRAINING_CFG = dotdict({
    'DATA': dotdict({
        'SEQ_LEN': 16,
        'STATE_LEN': 5,
        'SUBSAMPLE_FACTOR': 9,
        'IMSIZE': 64,
        }),
    'BATCH_SIZE': 1,
    })

FITVID_CFG = {
    'input_size': 3,
    'n_past': 5,
    'z_dim': 10,
    'beta': 1.e-4,
    'g_dim': 128,
    'rnn_size': 256,
    'num_channels': 64,
    }

BASE_FPS = 30
N_VIS_PER_BATCH = 1

def get_basedir():
    home = os.path.join(expanduser("~"), ".fitvid")
    if not os.path.exists(home):
        os.makedirs(home)
    return home

def download_from_s3(bucket, key, filepath):
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    s3.download_file(bucket, key, filepath)

def load_fitvid(pretrained=True):
    model = FitVid(**FITVID_CFG)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if pretrained:
        home = get_basedir()
        modelpath = os.path.join(home, "model.pt")
        if not os.path.exists(modelpath):
            download_from_s3('physion-physopt', '02bae92d9a4d4fa98537548ca80aa53a/artifacts/step_400000/model_ckpts/model.pt', modelpath)
        model.load_state_dict(torch.load(modelpath,
                                         map_location=torch.device(device)))
    return model


def download_test_data():
    # download example video
    home = get_basedir()
    hdf5path = os.path.join(home, 'physion_example.hdf5')
    download_from_s3('human-physics-benchmarking-towers-redyellow-pilot', 'pilot_towers_nb4_SJ025_mono1_dis0_occ0_tdwroom-redyellow_0011.hdf5', hdf5path)


def test_load_fitvid(): # test pretrained fitvid on example video
    download_test_data()

    # preprocess input
    images = []
    with h5py.File(hdf5path, 'r') as f:
        frames = list(f['frames'])
        img_transforms = transforms.Compose([
            transforms.Resize((PRETRAINING_CFG.DATA.IMSIZE, PRETRAINING_CFG.DATA.IMSIZE)),
            transforms.ToTensor(),
            ])
        for frame in frames[:PRETRAINING_CFG.DATA.SEQ_LEN*PRETRAINING_CFG.DATA.SUBSAMPLE_FACTOR:PRETRAINING_CFG.DATA.SUBSAMPLE_FACTOR]:
            img = f['frames'][frame]['images']['_img'][()]
            img = Image.open(io.BytesIO(img)) # (256, 256, 3)
            img = img_transforms(img)
            images.append(img)
        images = torch.stack(images, dim=0)
        images = torch.unsqueeze(images, 0) # add batch dim

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = load_fitvid().to(device)
    model.eval()
    with torch.no_grad():
        model_output = model(images.to(device))

    # save visualizations
    frames = {
        'gt': images,
        'sim': model_output['preds'].cpu().detach(),
        'stimulus_name': np.array(['example_video']),
        }
    save_vis(frames, PRETRAINING_CFG, home, 0)

def save_vis(input_frames, pretraining_cfg, output_dir, prefix=0, target_size=128):
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
