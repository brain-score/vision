import torch as ch

import shutil
try:
    import dill
except: # Hack to work with brainscore environment
    import pickle
    dill=pickle
import os
from subprocess import Popen, PIPE
import pandas as pd
from PIL import Image
from . import constants
try:
    from robustness.audio_functions import audio_transforms
except:
    print('Problem with audio transforms. This is fine if you are just using image models but audio models will not work.')

def has_attr(obj, k):
    """Checks both that obj.k exists and is not equal to None"""
    try:
        return (getattr(obj, k) is not None)
    except KeyError as e:
        return False
    except AttributeError as e:
        return False

def calc_est_grad(func, x, y, rad, num_samples):
    B, *_ = x.shape
    Q = num_samples//2
    N = len(x.shape) - 1
    with ch.no_grad():
        # Q * B * C * H * W
        extender = [1]*N
        queries = x.repeat(Q, *extender)
        noise = ch.randn_like(queries)
        norm = noise.view(B*Q, -1).norm(dim=-1).view(B*Q, *extender)
        noise = noise / norm
        noise = ch.cat([-noise, noise])
        queries = ch.cat([queries, queries])
        y_shape = [1] * (len(y.shape) - 1)
        l = func(queries + rad * noise, y.repeat(2*Q, *y_shape)).view(-1, *extender) 
        grad = (l.view(2*Q, B, *extender) * noise.view(2*Q, B, *noise.shape[1:])).mean(dim=0)
    return grad


def calc_fadein_eps(epoch, fadein_length, eps):
    """
    Calculate an epsilon by fading in from zero.

    Args:
        epoch (int) : current epoch of training.
        fadein_length (int) : number of epochs to fade in for.
        eps (float) : the final epsilon

    Returns:
        The correct epsilon for the current epoch, based on eps=0 and epoch
        zero and eps=eps at epoch :samp:`fadein_length` 
    """
    if fadein_length and fadein_length > 0:
        eps = eps * min(float(epoch) / fadein_length, 1)
    return eps

def ckpt_at_epoch(num):
    return '%s_%s' % (num, constants.CKPT_NAME)

def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k

        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes) 
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)

        Returns:
            A list of top-k accuracies.
    """
    with ch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [ch.round(ch.sigmoid(output)).eq(ch.round(target)).float().mean()], [-1.0] 

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].view(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact

class GraphPreprocessing(ch.nn.Module):
    '''
    Performs preprocessing of the input that should be included in the 
    graph for adversarial generation. Ie Input Normalization and Audio
    Representation Conversion.
    '''
    def __init__(self, dataset):
        super(GraphPreprocessing, self).__init__()
        self.normalize = InputNormalize(dataset.mean, dataset.std,
                                       dataset.min_value, dataset.max_value)
        if hasattr(dataset, 'audio_kwargs'):
            self.do_audio_preproc = True
            self.audio_preproc = AudioInputRepresentation(**dataset.audio_kwargs)# .cuda()
        else: 
            self.do_audio_preproc = False

    def forward(self, x):
        x = self.normalize(x)
        if self.do_audio_preproc:
            x = self.audio_preproc(x)
        return x
        

class InputNormalize(ch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified), clipped at the 
    possible min and max of the input
    '''
    def __init__(self, new_mean, new_std, min_value, max_value):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

        self.min_value=min_value
        self.max_value=max_value

    def forward(self, x):
        x = ch.clamp(x, self.min_value, self.max_value)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized


class AudioInputRepresentation(ch.nn.Module):
    '''
    A module (custom layer) for turning the audio signal into a 
    representation for training, ie using a mel spectrogram or a 
    cochleagram. 
    '''
    def __init__(self, rep_type, rep_kwargs, compression_type, compression_kwargs):
        super(AudioInputRepresentation, self).__init__()
        self.rep_type = rep_type
        self.rep_kwargs = rep_kwargs
        self.compression_type = compression_type
        self.compression_kwargs = compression_kwargs

        # Functions for the representations are defined in the audio_transforms 
        # library, but we only use the foreground audio here. 
        self.full_rep = audio_transforms.AudioToAudioRepresentation(rep_type,
                                                                    rep_kwargs, 
                                                                    compression_type,
                                                                    compression_kwargs)
    def forward(self, x): 
        x, _ = self.full_rep(x, None)
        return x


class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = ch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with ch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            ch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break

def save_checkpoint(state, is_best, filename):
    ch.save(state, filename, pickle_module=dill)
    if is_best:
        shutil.copyfile(filename, filename + constants.BEST_APPEND)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# ImageNet label mappings
def get_label_mapping(dataset_name, ranges):
    if dataset_name == 'imagenet':
        label_mapping = None
    elif dataset_name == 'restricted_imagenet':
        def label_mapping(classes, class_to_idx):
            return restricted_label_mapping(classes, class_to_idx, ranges=ranges)
    elif dataset_name == 'custom_imagenet':
        def label_mapping(classes, class_to_idx):
            return custom_label_mapping(classes, class_to_idx, ranges=ranges)
    else:
        raise ValueError('No such dataset_name %s' % dataset_name)

    return label_mapping

def restricted_label_mapping(classes, class_to_idx, ranges):
    range_sets = [
        set(range(s, e+1)) for s,e in ranges
    ]

    # add wildcard
    # range_sets.append(set(range(0, 1002)))
    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(range_sets):
            if idx in range_set:
                mapping[class_name] = new_idx
        # assert class_name in mapping
    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping

def custom_label_mapping(classes, class_to_idx, ranges):

    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(ranges):
            if idx in range_set:
                mapping[class_name] = new_idx

    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping
