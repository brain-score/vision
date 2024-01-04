import time
import torch
import torch.nn as nn
from torchvision import transforms as T
import numpy as np
from PIL import Image
from turbojpeg import TurboJPEG, TJPF, TJSAMP
from urllib.request import urlopen
import matplotlib.pyplot as plt
import cv2
from IPython.core.debugger import set_trace
from .colormap import colormap

try:
    from fastprogress.fastprogress import progress_bar
except:
    from fastprogress import progress_bar

turbo = TurboJPEG()

color_list = colormap()
_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

# __all__ = ['turbo_loader', 'pil_loader', 'open_image', 'load_image_rgb']

TEST_IMAGE_URL = 'https://scorsese.wjh.harvard.edu/turk/stimuli/turbo/butterfly.jpg'

def turbo_loader(file, to_rgb=True):
    with open(file, 'rb') as f:
        # have to install latest to access crop features:
        # buf = turbo.crop(f.read(), x=0, y=0, w=100, h=100, preserve=False, gray=True)
        img = turbo.decode(f.read(), pixel_format=TJPF.RGB if to_rgb else TJPF.BGR)
    return img

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def cv2_loader(path, to_rgb=True):
    img = cv2.imread(path)
    if to_rgb: img = img[:,:,::-1]
    
    return img

def open_image(p, to_rgb=True):
    '''Our default image loader, takes `filename` and returns a PIL Image. 
        Speedwise, turbo_loader > pil_loader > cv2, but cv2 is the most robust, so 
        we try to load jpg images with turbo_loader, fall back to PIL, then cv2.
        
        This fallback behavior is needed, e.g., with ImageNet there are a few images
        that either aren't JPEGs or have issues that turbo_loader crashes on, but cv2 
        doesn't.
    '''
    if p.lower().endswith('.jpg') or p.lower().endswith('.jpeg'): 
        try:
            img = turbo_loader(p, to_rgb=to_rgb)
        except:
            try:
                img = pil_loader(p)
            except:
                img = cv2.imread(p)
                if to_rgb: img = img[:,:,::-1]
    else:
        try:
            img = pil_loader(p)
        except:
            img = cv2.imread(p)
            if to_rgb: img = img[:,:,::-1]
                
    if img is not None and not isinstance(img, Image.Image):
        img = Image.fromarray(img)
        
    return img

def open_image_array(p, to_rgb=True):
    '''Our default image loader, takes `filename` and returns a PIL Image. 
        Speedwise, turbo_loader > pil_loader > cv2, but cv2 is the most robust, so 
        we try to load jpg images with turbo_loader, fall back to PIL, then cv2.
        
        This fallback behavior is needed, e.g., with ImageNet there are a few images
        that either aren't JPEGs or have issues that turbo_loader crashes on, but cv2 
        doesn't.
    '''
    if p.lower().endswith('.jpg') or p.lower().endswith('.jpeg'): 
        try:
            img = turbo_loader(p, to_rgb=to_rgb)
        except:
            try:
                img = np.array(pil_loader(p))
            except:
                img = cv2.imread(p)
                if to_rgb: img = img[:,:,::-1]
    else:
        try:
            img = pil_loader(p)
        except:
            img = cv2.imread(p)
            if to_rgb: img = img[:,:,::-1]                
        
    return img

def download_image(url):
    data = urlopen(url).read()
    data = np.frombuffer(data, np.uint8)
    img = turbo.decode(data, pixel_format=TJPF.RGB)
    img = Image.fromarray(img)
    return img

def hasattrs(o,attrs):
    "from fastai2/torch_core.py, Test whether `o` contains all `attrs`"
    return all(hasattr(o,attr) for attr in attrs)

def _fig_bounds(x):
    '''from fastai2/torch_core.py'''
    r = x//32
    return min(5, max(1,r))

def show_image(im, ax=None, figsize=None, title=None, ctx=None, stats=None, **kwargs):
    "Show a PIL or PyTorch image on `ax`. modified from fastai2/torch_core.py"
    
    # Handle pytorch axis order
    if hasattrs(im, ('data','cpu','permute')):
        im = im.data.cpu()
        if im.shape[0]<5: 
            if stats is not None and stats['mean'] is not None and stats['std'] is not None:
                # normalize = T.Normalize(mean.tolist(), std)
                mean, std = np.array(stats['mean']), np.array(stats['std'])
                invnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
                im = invnormalize(im)
            im=im.permute(1,2,0)
        
    elif not isinstance(im,np.ndarray): 
        im = np.array(im)
    elif isinstance(im, np.ndarray):
        if im.shape[0]<5: im = np.moveaxis(im, 0, -1)
            
    # Handle 1-channel images
    if im.shape[-1]==1: im=im[...,0]

    ax = ctx if ax is None else ax
    if figsize is None: 
        figsize = (_fig_bounds(im.shape[0]), _fig_bounds(im.shape[1]))
    if ax is None: 
        _,ax = plt.subplots(figsize=figsize)
        
    ax.imshow(im, **kwargs)
    if title is not None: ax.set_title(title)
    ax.axis('off')
    
    return ax;

def show_batch(b, nrows=1, ncols=None, height=2.0, width=16, title='ImageBatch',
               fmt='', vals=None, stats=None):
    ncols = len(b) if ncols is None else ncols
    fig,axs = plt.subplots(nrows, ncols, figsize=(width,height*nrows))
    fig.suptitle(title)
    for i,ax in enumerate(axs.flatten()):
        subtitle = fmt.format(vals[i]) if vals is not None else ''
        show_image(b[i], ctx=ax, title=subtitle, stats=stats)
        
def show_grid(imgs, idx, stats): 
    if stats is not None:
        mean, std = np.array(stats['mean']), np.array(stats['std'])
        inorm = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        img_list = [inorm(batch[idx]).permute(1,2,0) for batch in imgs]
        return Image.fromarray(np.concatenate([np.array(img.cpu()*255).astype(np.uint8) for img in img_list],axis=1))
    else:
        img_list = [batch[idx] for batch in imgs]
        return Image.fromarray(np.concatenate([np.array(img).astype(np.uint8) for img in img_list],axis=1))
        
def show_act_grid(imgs, idx):
    img_list = [batch[idx] for batch in imgs]
    nrows, ncols = img_list[0].shape[0], len(img_list)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True, figsize=(ncols*4,nrows*4))
    for row,axes in enumerate(axs):
        for col,ax in enumerate(axes):
            ax.imshow(img_list[col][row].cpu(), extent=[0, 1, 0, 1], cmap='gray')
            ax.axis('off')
            ax.axis('tight')
            ax.axis('image')    

def test_loader(dataloader, after_batch=None):
    c = 0
    start = time.time()
    indexes = None
    for batch_num, batch in enumerate(progress_bar(dataloader)):
        if after_batch is not None:
            batch = after_batch(batch)
        c += len(batch[-1]) if isinstance(batch[-1], list) else batch[-1].shape[0]
    dur = time.time() - start
    fps = c / dur
    #print("\n",dur, c, fps)
    print("\n{:d}images, {:4.0f}s, {:4.0f}imgs/s".format(c,dur,fps))
    return batch        

def draw_bbox(img, bbox, color, pen_width=1):
    """add a bounding box [x_min, y_min, x_max, y_max] with values 0-1 to an image"""
    was_pil = isinstance(img, (Image.Image))
    img = np.array(img)
    h,w,_ = img.shape
    x_min,y_min,x_max,y_max = int(bbox[0]*w),int(bbox[1]*h),int(bbox[2]*w),int(bbox[3]*h)
    col = tuple(int(c) for c in color)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), col, pen_width)
    return Image.fromarray(img) if was_pil else img

def draw_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=0):
    """Visualizes a single binary mask."""
    
    was_pil = isinstance(img, (Image.Image))
    img = np.array(img)
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if border_thick:
        contours, hierarchy = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)
    
    img = img.astype(np.uint8)
    
    return Image.fromarray(img) if was_pil else img