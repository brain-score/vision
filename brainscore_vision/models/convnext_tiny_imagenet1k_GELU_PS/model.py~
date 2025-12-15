from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.check_submission import check_models
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import wget
from urllib.request import urlretrieve
from pathlib import Path

###################################################################################################################################

def get_model(name):
    assert name == 'convnext_tiny_imagenet1k_GELU_crop'
    fileName='ConvNeXt-tiny_ImageNet_None_None_None_GELU_Conv_LayerNorm_StridedConv_CELoss_None'
    netType, task, dataOOD, dataAug, preprocFunc, actFunc, intFunc, normFunc, poolFunc, lossFunc, balanceMethod = fileName.split('_')
    imDims=np.array([3, 224, 224]) #
    numClasses=1000 #

    #device=torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    fileName = f'{fileName}.pt'
    weightURL = f'https://codeberg.org/mwspratling/HAND/raw/branch/main/{fileName}'
    fileName = wget.download(weightURL, fileName)
    
    model = define_model(netType, preprocFunc, actFunc, intFunc, normFunc, poolFunc, imDims, numClasses, task, device)
    #model = import_weights('brainscore_vision/models/convnext_tiny_imagenet1k_GELU_crop/'+fileName+'.pt', model, device)
    model = import_weights(fileName, model, device)

    preprocessing = functools.partial(load_preprocess_images, image_size=imDims[-1],
                                      normalize_mean=(0, 0, 0), normalize_std=(1, 1, 1))
    wrapper = PytorchWrapper(identifier='convnext_tiny_imagenet1k_GELU_crop', model=model, preprocessing=preprocessing)
    wrapper.image_size = imDims[-1]
    return wrapper


def get_layers(name):
    assert name == 'convnext_tiny_imagenet1k_GELU_crop'
    #return ['inputPreproc', 'downsample_layers.0', 'downsample_layers.1', 'downsample_layers.2', 'downsample_layers.3','stages.0.0.act', 'stages.0.1.act', 'stages.0.2.act','stages.1.0.act', 'stages.1.1.act', 'stages.1.2.act','stages.2.0.act', 'stages.2.1.act', 'stages.2.2.act', 'stages.2.3.act', 'stages.2.4.act', 'stages.2.5.act', 'stages.2.6.act', 'stages.2.7.act', 'stages.2.8.act','stages.3.0.act', 'stages.3.1.act', 'stages.3.2.act','head'] #ConvNeXt
    return ['downsample_layers.0', 'downsample_layers.1', 'downsample_layers.2', 'downsample_layers.3', 'stages.0.1.act', 'stages.1.1.act', 'stages.2.1.act', 'stages.2.3.act', 'stages.2.5.act', 'stages.2.7.act', 'stages.3.1.act', 'head'] #ConvNeXt


def get_bibtex(model_identifier):
    return """"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)

###################################################################################################################################
from PIL import Image
def load_preprocess_images(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images

def load_images(image_filepaths):
    return [load_image(image_filepath) for image_filepath in image_filepaths]

def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper() \
                and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image

def preprocess_images(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images

def torchvision_preprocess_input(image_size, **kwargs):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.Resize((image_size, image_size)),
        torchvision_preprocess(**kwargs),
    ])

def torchvision_preprocess(normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])

#############################################################################
# Code that defines neural network architectures
#
# (c) 2023 Michael William Spratling
#############################################################################

def define_model(netTypeAndAug, preprocFunc, actFunc, intFunc, normFunc, poolFunc,
                 imDims, numClasses, task, device):
    if poolFunc.startswith('Channel'): # determine number of channels used for channel-wise pooling
        poolFunc,sizeChannelPool=poolFunc.split('Pool')
        sizeChannelPool=int(sizeChannelPool)
    else:
        sizeChannelPool=1

    netTypeAndAug=netTypeAndAug.split('+')
    # create a model that is defined using this codebase
    netType=netTypeAndAug[0] #define the base model: ignore any part of the name of the model that comes after a plus sign
    if netType.startswith('Ensemble'):
        model = Ensemble(netType, preprocFunc, actFunc, intFunc, normFunc,
                         poolFunc+'Pool'+str(sizeChannelPool), imDims, numClasses, task,
                         device)


    if netType.startswith('ConvNeXt'):
        version=parse_string(netType,'ConvNeXt')[1]
        if len(version)==0:
            version='base' #default value
        if version=='tiny':
            depths=[3, 3, 9, 3]
            dims=[96, 192, 384, 768]
        elif version=='small':
            depths=[3, 3, 27, 3]
            dims=[96, 192, 384, 768]
        elif version=='base':
            depths=[3, 3, 27, 3]
            dims=[128, 256, 512, 1024]
        elif version=='large':
            depths=[3, 3, 27, 3]
            dims=[192, 384, 768, 1536]
        elif version=='xlarge':
            depths=[3, 3, 27, 3]
            dims=[256, 512, 1024, 2048]
        model = ConvNeXt(in_chans=imDims[0], num_classes=int(numClasses), depths=depths,
                         dims=dims, preprocFunc=preprocFunc, actFunc=actFunc)

    model.to(device)
    return model

#############################################################################
# ConvNeXt - code modified from:
# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
#############################################################################
from timm.layers import trunc_normal_, DropPath

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", affine=True):
        super().__init__()
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
        self.affine=affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.affine: x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BlockConvNeXtConv(nn.Module):
    r""" ConvNeXt Block. Implemented using 1x1 conv instead of linear layers
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,
                 actFunc='GELU', bias=True, affine=True):
        super().__init__()

        if actFunc=='HAND':
            #modify the standard implementation for new activation function
            bias=False
            affine=False

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=bias) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first", affine=affine)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=bias)
        self.act = define_activation_function(actFunc, 4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1,dim,1,1)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., preprocFunc='None',
                 actFunc='GELU'):
        super().__init__()
        self.inputPreproc=define_preproc_function(preprocFunc, in_chans)

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[BlockConvNeXtConv(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value, actFunc=actFunc) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, lossFunc='', labels=None, numClasses=None, yAllReqd=False, yAllTarget=None, warpMatrix=None):
        loss=torch.tensor([0.0], device=x.device)
        yAll=[]
        x = self.inputPreproc(x)

        x = self.forward_features(x)
        x = self.head(x)

        if yAllReqd:
            return yAll #,loss
        elif self.training:
            return x,loss
        else:
            return x



#############################################################################
# Code defining pre-processing functions
#
# (c) 2025 Michael William Spratling
#############################################################################
def define_preproc_function(preprocFunc, numImageChannels):
    #convert string defining name of function to a definition for a neural net module
    if preprocFunc=='None' or preprocFunc=='Identity':
        preprocFunc=nn.Identity()
    else:
        print("ERROR: undefined preproc function")
    return preprocFunc



#############################################################################
# Code defining activation functions
#
# (c) 2023 Michael William Spratling
#############################################################################
def define_activation_function(actFunc, numChannels=None):
    #convert string defining name of function to a definition for a neural net module
    if actFunc=='None' or actFunc=='Identity':
        actFunc=nn.Identity()
    elif actFunc=='ReLU':
        actFunc=nn.ReLU()
    elif actFunc=='ReLU6':
        actFunc=nn.ReLU6()
    elif actFunc=='LReLU':
        actFunc=nn.LeakyReLU()
    elif actFunc=='PReLU':
        actFunc=nn.PReLU()
    elif actFunc=='ELU':
        actFunc=nn.ELU()
    elif actFunc=='SiLU':
        actFunc=nn.SiLU()
    elif actFunc=='SELU':
        actFunc=nn.SELU()
    elif actFunc=='GELU':
        actFunc=nn.GELU()
    elif actFunc=='tanh':
        actFunc=nn.Tanh()
    elif actFunc=='Mish':
        actFunc=nn.Mish()
    elif actFunc=='Sigmoid':
        actFunc=nn.Sigmoid()
    elif actFunc=='HAND':
        actFunc=HAND(numChannels)
    else:
        print("ERROR: undefined activation function")
    return actFunc

class HAND(nn.Module):
    # Homeostasis -> Accelerating Nonlinearity -> Divisive-nomalisation
    def __init__(self, numChannels, scale=2):
        super(HAND,self).__init__()
        self.xNorm = nn.BatchNorm2d(numChannels, affine=False)
        self.scale = scale
        self.logexponent = nn.Parameter(torch.zeros(1))
        self.logsemisaturation = nn.Parameter(torch.zeros(1))
        self.logpoolWeight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # leant params are constrained to be non-negative
        exponent = torch.exp(self.logexponent) + 1
        poolWeight = torch.exp(self.logpoolWeight) #+ 1
        semisaturation = torch.exp(self.logsemisaturation) #+ 1

        # Homeostasis (standardise input stats to prevent dead units)
        x = self.xNorm(x)

        # Accelerating Nonlinearity
        y = F.relu(x)**exponent

        # Divisive-normalisation (competition between neurons with same RF
        #                         i.e. across columns)
        yCompPool =  torch.mean(y, dim=1, keepdim=True)
        z = y / (semisaturation + poolWeight*yCompPool)
        z *= self.scale
        return z


#############################################################################
# Code defining integration functions
#
# (c) 2023 Michael William Spratling
#############################################################################
def define_integration_function(intFunc, inChannels, outChannels, kernel_size, stride, padding, bias, layerDims=None, depth=None, numLayers=None, sizeChannelPool=None):
    #convert string defining name of function to a definition for a neural net module
    if intFunc=='Conv':
        intFunc=nn.Conv2d(inChannels, outChannels, kernel_size, stride, padding, bias=bias)
    elif intFunc=='ConvBias':
        intFunc=nn.Conv2d(inChannels, outChannels, kernel_size, stride, padding, bias=True) #over-ride model def to force conv layer to have bias term
    elif intFunc=='ConvNobias':
        intFunc=nn.Conv2d(inChannels, outChannels, kernel_size, stride, padding, bias=False) #over-ride model def to force conv layer to have no bias term
    else:
        print("ERROR: undefined integration function")
    return intFunc


#############################################################################
# Code defining pooling functions
#
# (c) 2023 Michael William Spratling
#############################################################################
def define_pooling_function(poolFunc, poolSize, inChannels=None, outChannels=None, stride=None):
    #convert string defining name of function to a definition for a neural net module
    if poolFunc=='AvgPool':
        poolFunc=nn.AvgPool2d(poolSize, stride=stride)
    elif poolFunc=='MaxPool':
        poolFunc=nn.MaxPool2d(poolSize, stride=stride)
    elif poolFunc.startswith('LPPool'):
        normVal=parse_string(poolFunc,'LPPool','/')
        normVal=float(normVal[0])
        poolFunc=nn.LPPool2d(normVal, poolSize, stride=stride)
    else:
        print("ERROR: undefined pooling function")
    return poolFunc

#############################################################################
#  Code defining norm functions
#
# (c) 2023 Michael William Spratling
#############################################################################
def define_norm_function(normFunc, numChannels):
    #convert string defining name of function to a definition for a neural net module
    if normFunc.startswith('Identity') or normFunc=='none':
        normFunc=nn.Identity()
    elif normFunc=='BatchNorm':
        normFunc=nn.BatchNorm2d(numChannels)
    elif normFunc=='BatchNormNonAffine':
        normFunc=nn.BatchNorm2d(numChannels,affine=False)
    elif normFunc=='BatchStandardise':
        normFunc=BatchStandardise(numChannels)
    elif normFunc=='InstanceNorm':
        normFunc=nn.InstanceNorm2d(numChannels)
    elif normFunc=='InstanceNormAffine':
        normFunc=nn.InstanceNorm2d(numChannels,affine=True) #,eps=0.2)
    elif normFunc=='LayerNorm':
        normFunc=nn.LayerNorm([numChannels,None,None])
    elif normFunc=='LocalResponseNorm':
        normFunc=nn.LocalResponseNorm(numChannels,alpha=0.1, beta=0.5, k=1.0)
    else:
        print("ERROR: undefined norm function")
    return normFunc

#############################################################################
# Code defining methods for updating weights
# (loss functions, regularisation methods and learning rules)
#
# (c) 2023 Michael William Spratling
#############################################################################
class classification_loss_class(nn.Module):
    def __init__(self, lossFunc, device, params, paramsVariable=False):
        super(classification_loss_class, self).__init__()
        self.lossFunc=lossFunc
        self.device=device
        self.params=params
        self.paramsVariable=paramsVariable
    def forward(self, y, labels):
        loss=classification_loss_func(y, labels, self.lossFunc, self.device, self.params)
        return loss

def classification_loss_func(y, labels, lossFunc, device, params=None, weightPerClass=None, progress=None, targets=None):
    # loss function for classification/prediction accuracy
    numClasses=y.size(1)
    if targets==None:
        # convert labels to a one-hot encoding
        targets = F.one_hot(labels, num_classes=numClasses).type(torch.LongTensor).to(device)
    if y.ndim>2: #assume this is a segmentation task
        #re-organise data so that dimension 0 contains all samples (i.e. B*H*W) 
        #and dimension 1 contains the class information.
        #for targets convert (B,H,W,C) to (B*H*W,C)
        targets = targets.view(torch.prod(torch.tensor(targets.size())[:-1]), numClasses) #gather B, H, and W into 1st dimension
        #for y convert (B,C,H,W) to (B*H*W,C)
        y = y.permute(0,2,3,1).contiguous() # move class to last dimension
        y = y.view(torch.prod(torch.tensor(y.size())[:-1]), numClasses) #gather B, H, and W into 1st dimension
        #remove samples where there is no target (i.e. where there is no labelled object, or pixel is to be ignored)
        #targetExists=torch.sum(targets,dim=1)>0
        #targets=targets[targetExists,:]
        #y=y[targetExists,:]
        #assume that losses that use labels can also cope with being supplied with targets (true for CE loss)
        labels=targets.type(torch.float)

    loss=torch.tensor([0.0], device=device)
    if lossFunc.startswith('CELoss'): #=Cross-Entropy Loss=NLL after log_softmax activation
        if params!=None:
            y=y/params #apply temperature value
        #loss=-torch.mean(torch.sum(targets*F.log_softmax(y,-1), dim=-1))
        loss=F.cross_entropy(y, labels, weight=weightPerClass)
    elif lossFunc.startswith('NLLLoss'): #=negative log likelihood loss (assumes activities have been normalised)
        loss=F.nll_loss(y, labels, weight=weightPerClass)
    elif lossFunc.startswith('NormNLLLoss'): #=negative log likelihood loss following normalisation by the Lp-norm
        if params!=None:
            y=y/(1e-6+torch.norm(y, p=params, dim=-1, keepdim=True))
        else:
            y=y/(1e-6+torch.norm(y, p=1, dim=-1, keepdim=True))
        loss=F.nll_loss(y, labels, weight=weightPerClass)
    elif lossFunc.startswith('SELoss'): #mean Squared Error Loss
        #loss=torch.mean(torch.mean((targets-y)**2,dim=-1))
        loss=F.mse_loss(y,targets)
    elif lossFunc.startswith('SEwsLoss'): #a variation on SELoss that takes the sum of the errors weighted by the errors themselves
        loss=squared_error_loss(y,targets, reduction='weightedSum')
    elif lossFunc.startswith('AELoss'): #=mean Absolute Error Loss
        #loss = torch.mean(torch.mean(torch.abs(targets-y),dim=-1))
        loss=F.l1_loss(y,targets)
    elif lossFunc.startswith('FocalLoss'): #=Focal loss
        loss=focal_loss(y, labels, targets, weightPerClass=weightPerClass, gamma=params)
    elif lossFunc.startswith('NCSLoss'): #
        loss=negative_cosine_similarity_loss(y, targets)
    elif lossFunc.startswith('DICELoss'):
        loss=dice_loss(y, targets)

    elif lossFunc.startswith('LNLoss'): #=Logit Norm loss, see https://arxiv.org/abs/2205.09310
        loss=logit_norm_loss(y, labels, weightPerClass=weightPerClass, temp=params)
    elif lossFunc.startswith('LNaLoss'): #=Logit Norm with adaptive temp
        loss=logit_norm_adaptive_loss(y, targets, weightPerClass=weightPerClass, tempPow=params)
    elif lossFunc.startswith('LALoss'): #=Logit adjusted loss, see https://arxiv.org/abs/2007.07314
        #calculate prior probability for each class: this code assumes that weightPerClass has been set using balanceMethod='InvNum1.0'
        #note that if all classes have the same probability (i.e. if weightPerClass is uniform) then this loss is identical to CE
        #note that the loss component of BALMS is implemented identically (see https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification/blob/main/loss/BalancedSoftmaxLoss.py)
        probPerClass=1/weightPerClass 
        probPerClass/=torch.sum(probPerClass)
        adjlogits=y+torch.log(1e-12+probPerClass)
        loss=F.cross_entropy(adjlogits, labels)

    elif lossFunc.startswith('MMLoss'): #=multi-class margin loss=Hinge loss
        #loss=F.multi_margin_loss(y, labels, weight=weightPerClass, margin=params)
        loss=multi_margin_loss(y, targets, weightPerClass=weightPerClass, sepMargin=params)
    return loss

#############################################################################
# Code for various generally useful functions
#
# (c) 2023 Michael William Spratling
#############################################################################
def get_num_devices():
    numDevices=int(np.maximum(1,torch.cuda.device_count()))
    if torch.cuda.is_available() and 'AMD' in torch.cuda.get_device_name(0):
        numDevices=1 #ROCm not compatible with DataParallel so just use one device
    return numDevices

def get_max_runtime(runTimeLimitDefault):
    runTimeLimit=None
    job_id = os.getenv('SLURM_JOB_ID') # find current job's ID
    if job_id is not None:
        try:
            # run the scontrol command to get job details
            result = subprocess.run(['scontrol', 'show', 'job', str(job_id)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, text=True)
            # find line in scontrol's output specifying the TimeLimit
            for line in result.stdout.split('\n'):
                if 'TimeLimit=' in line:
                    # parse the TimeLimit to calculate job's maximum runtime in seconds
                    runTimeLimitStr = line.split('TimeLimit=')[1].split()[0]
                    try:
                        days,hours=runTimeLimitStr.split('-')
                        runTimeLimit=int(days)*24*60*60
                    except:
                        hours=runTimeLimitStr
                        runTimeLimit=0
                    hours,minutes,seconds=hours.split(':')
                    runTimeLimit+=int(hours)*60*60+int(minutes)*60+int(seconds)
                    runTimeLimit=float(runTimeLimit)
        except:
            pass
    if runTimeLimit==None:
        runTimeLimit=runTimeLimitDefault
        print('WARNING: failed to find job run-time limit, using default', runTimeLimit)
    else:
        print('runTimeLimit=', runTimeLimit)
    return runTimeLimit

def elapsed_time(timeStart=None, name='time'):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    timeCurrent=time.time()
    if timeStart != None:
        timeElapsed = timeCurrent - timeStart
        print(name+' {:.4f}s'.format(timeElapsed), flush=True)
    else:
        timeElapsed=None
    return timeCurrent, timeElapsed

def make_odd(i):
    #round number to the next highest odd integer value
    i=np.ceil(i)
    isEven=(i % 2 == 0)
    i+=isEven
    return i

def round_to_odd(i):
    #round number to the nearest odd integer value
    nearest=round(i)
    if (nearest % 2 == 0):
        # nearest is even so choose the next nearest integer
        if nearest<=i: nearest+=1
        elif nearest>i: nearest-=1
    return nearest

def round_to_even(i):
    #round number to the nearest even integer value
    nearest=round(i)
    if (nearest % 2 != 0):
        # nearest is odd so choose the next nearest integer
        if nearest<=i: nearest+=1
        elif nearest>i: nearest-=1
    return nearest

def num_clip(x, min=-float('inf'), max=float('inf')):
    #clips a vaue to be in the range min and max, like torch.clamp but for python scalar values
    if x < min: x=min
    if x > max: x=max
    return x

def calc_CNN_layer_sizes(imDims,numMasks,maskSize,padding,stride,poolSize):
    mapDims=[]
    newDims=imDims[1:]
    for i in range(np.size(numMasks)):
        newDims=1+(newDims-maskSize[i]+2*padding[i])/stride[i] #W & H of conv layer
        newDims=newDims/poolSize[i] #W & H of pooling layer (assumes no padding or dilation, and stride equal to pooling area)
        mapDims.append(np.array(newDims))
    mapDims= np.asarray(mapDims) #convert a list of 1D lists into a 2D array
    print(mapDims)
    return mapDims

def center_coords(activations):
    center=torch.tensor(activations.size())[-2:]/2
    center=int(center[0]),int(center[1])
    return center

def print_model_details(model):
    print(model)
    #params=list(model.parameters())
    #for layer in range(len(params)):
    #    print(params[layer].size()) 
    print('Total Parameters =',sum(p.numel() for p in model.parameters()))
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Trainable Parameters =',params)
    print('\n',flush=True)
    return params

def test_network_accuracy(numClasses, test_loader, model, detailed=True):
    #test and report the classification accuracy of a network
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    # initialize variables to record metrics
    classCorrect = torch.zeros(numClasses, device=device)
    classTotal = torch.zeros(numClasses, device=device)
    # present samples to network and record performance
    for samples, labels in test_loader:
        samples, labels = samples.to(device), labels.to(device)

        #samples=random_saturation(samples, prob=1, maxChange=1.0)
        #samples=random_contrast(samples, prob=1, maxChange=0.5)
        #samples=random_brightness(samples, prob=1, maxChange=0.5)
        #samples=random_sharpness(samples, prob=1, maxChange=1.0)
        #samples=random_gaussian_blur(samples, prob=1, sigmaMax=0.06)
        #samples=mix_random_noise(samples, prob=1, noiseMax=0.25)
        #sample=random_erase(samples, prob=1, scale=(0.0, 0.125), ratio=5, value='uniform')
        #samples,_=random_perspective(samples, warpMag=0.05, padding='scramble')
        #plot_images(samples, labels)

        # get model predictions and compare to ground-truth
        with torch.no_grad(): y = model(samples)
        _, pred = torch.max(y, dim=1)
        correct = labels==pred

        targets = F.one_hot(labels.to(torch.int64), num_classes=numClasses)
        classTotal += torch.sum(targets,dim=0)
        classCorrect += torch.sum(targets*correct[:,None],dim=0)
        
        #record accuracy separately for each class
        # for i in range(samples.size(0)):
        #     label = labels[i]
        #     classCorrect[label] += correct[i]
        #     classTotal[label] += 1

    if detailed:
        # calculate and print test accuracies for individual classes
        for i in range(numClasses):
            if classTotal[i] > 0:
                print('Accuracy for class %d: %2d%% (%2d/%2d)' % (
                    i, 100 * classCorrect[i] / classTotal[i],
                    torch.sum(classCorrect[i]), torch.sum(classTotal[i])))
            else:
                print('Accuracy for class %d: N/A (no training examples)' % i)
    # calculate and print overall accuracy
    accuracyOverall=100. * torch.sum(classCorrect) / torch.sum(classTotal)
    print('Accuracy (for all classes): %.2f %% (%2d/%2d)\n' % (
        accuracyOverall, torch.sum(classCorrect), torch.sum(classTotal)),flush=True)
    return accuracyOverall

def quantise(x, levels=10, slope=10.0):
    #y=torch.round(x*levels)*(1.0/levels)
    y=torch.zeros(x.size(), device=x.device)
    for i in range(levels):
        y+=torch.sigmoid(slope*(x-(i+0.5)/(levels))) 
    y/=levels
    return y

def population_code(x, levels=10, slope=200.0):
    #expands number of channels (dim 1) by population coding the input, x. Assumes values of x are in range [0,1] 
    y=torch.tensor([], device=x.device)
    for i in range(levels):
        vals=torch.exp(-0.5*slope*((x-(i+0.5)/(levels))**2))
        #vals=torch.sigmoid(slope*(x-(i+0.5)/(levels)))
        y=torch.cat((y,vals),dim=1)
    return y

def sigmoid(x, offset=0, slope=1):
    #a wrapper for the torch sigmoid function that allows for convientient definition of offset and slope
    y=F.sigmoid(slope*(x-offset))
    return y
    
import shutil
from os.path import exists
def save_checkpoint(fileName, model, optimizer, epoch):
    #first make copy of existing checkpoint, in case job times-out during save
    if exists(fileName):
        shutil.copy(fileName,fileName+'_bckup')
    #create checkpoint data structure and save it
    state={
        'model_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, fileName)

def load_checkpoint(fileName, model, optimizer, device):
    print("Loading checkpoint. If this generates an error, it is because file is not a checkpoint: delete %s and re-run train_model.py to train a new model from scratch." % fileName)
    state = torch.load(fileName,map_location=device)
    epoch = state['epoch']
    model.load_state_dict(deparallalise_state_dict(state['model_dict']))
    model.to(device) #should cause optimizer state to be cast to correct device in next line of code
    optimizer.load_state_dict(deparallalise_state_dict(state['optim_dict']))
    print('RESUMING TRAINING from epoch',epoch,flush=True)
    return model, optimizer, epoch

from collections import OrderedDict
def deparallalise_state_dict(state_dict):
    #if model was trained using DataParallel, then the name “module.” was appended to each item, 
    #and needs to be removed to allow loading to a non-DataParallel model
    new_state_dict = OrderedDict()
    for name, val in state_dict.items():
        if name[:7] == 'module.':
            name = name[7:]  # remove `module.`
        new_state_dict[name] = val
    return new_state_dict

# def parallel_model(model, state, device):    
#     for key in state.keys():
#         break
#     if key.startswith('module'):
#         model=nn.DataParallel(model).to(device)
#     return model

def import_weights(fileName, model, device):
    if Path(fileName).is_file():
        #import just the parameters from a saved checkpoint/state_dict
        state = torch.load(fileName,map_location=device)
        if 'state_dict' in state.keys():
            state=state['state_dict']
        #model = parallel_model(model, state, device)   
        model.load_state_dict(deparallalise_state_dict(state))
        if 'inputNorm.inputMean' in state.keys():
            model.inputNorm.inputMean=nn.Parameter(state['inputNorm.inputMean'], requires_grad=False)
            model.inputNorm.inputStd=nn.Parameter(state['inputNorm.inputStd'], requires_grad=False)
        model.to(device) 
        print('Imported weights from',fileName)
        if 'epoch' in state.keys():
            print('saved at epoch',state['epoch'],flush=True)
    else:
        print('***WARNING***: Tried to import weights from non-existent file:',fileName)
    #print_model_details(model)
    return model
    
def pasteable_array(vals, decPlaces=6):
    #print values on screen with tab seperators, so that they can be copy-and-pasted into a spreadsheet
    pasteable='\t'.join([str(round(v,decPlaces)) for v in vals.numpy()])
    print(pasteable)  

def print_stats(vals,name=''):
    if vals.numel()==1:
        if vals.abs().max()<0.001: print('%s \t%+.2e' % (name,vals.item()))
        else: print('%s \t%+.4f' % (name,vals.item()))
    elif vals.numel()>1:
        if vals.abs().max()<0.001:
            print('%s \t%+.2e \t%+.2e \t%+.2e \t%+.2e \t%.2e \t%s' % (name,vals.min().item(),vals.max().item(),vals.float().mean().item(),vals.abs().float().mean().item(),vals.float().std(correction=0).item(),str(vals.size()).split('Size')[1]))
        else:
            print('%s \t%+.4f \t%+.4f \t%+.4f \t%+.4f \t%.4f  \t%s' % (name,vals.min().item(),vals.max().item(),vals.float().mean().item(),vals.abs().float().mean().item(),vals.float().std(correction=0).item(),str(vals.size()).split('Size')[1]))

def plot_image(I, incTicks=False, colourbar=True):
    if I.size(0)==1:
        plt.imshow(I[0], cmap='gray_r', interpolation='none')
    elif I.dim()==2:
        plt.imshow(I, cmap='gray_r', interpolation='none')
    else:
        I-=min_values(I, keepdim=True)
        I/=max_values(I, keepdim=True)
        I=np.transpose(I, [1,2,0])
        plt.imshow(I) #, cmap='RGB')
        
    if not incTicks:
        plt.xticks([])
        plt.yticks([])
    if colourbar: plt.colorbar()
 
def plot_images(samples, labels=None, numToPlot=20):
    samples=samples.detach().to(dtype=torch.float32).to(torch.device("cpu"))
    numToPlot=np.minimum(numToPlot,samples.size(0))
    numRows=int(np.sqrt(numToPlot))
    numCols=np.maximum(1,int(numToPlot/numRows))
    fig = plt.figure(figsize=(numCols, numRows))
    numToPlot=np.minimum(numToPlot,numRows*numCols)
    for idx in range(numToPlot):
        plt.subplot(numRows, numCols, idx+1)
        #plt.tight_layout()
        if idx==numCols*(numRows-1):
            #to avoid over-crowding, put ticks only on the bottom-left subfigure
            incTicks=True
        else:
            incTicks=False
        plot_image(samples[idx], incTicks)
        if labels!=None:
            if labels[idx]==labels[idx]//1:
                plt.title("class {}".format(labels[idx]))
            else:
                plt.title("{:.3f}".format(labels[idx]))

    fig
  
def plot_weights(W, imDims, numToPlot=20, figTitle=None):
    W=W.detach().to(dtype=torch.float32).to(torch.device("cpu"))
    numToPlot=np.minimum(numToPlot,W.size(0))
    numRows=2
    numCols=np.maximum(1,int(numToPlot/numRows))
    fig = plt.figure(figsize=(numCols, numRows))
    for idx in np.arange(numToPlot):
        fig.add_subplot(numRows, numCols, idx+1)
        w=W[idx,:]
        if imDims[0]==1:
            w=w.reshape(imDims[1],imDims[2])
        else:
            w=w.reshape(imDims[0],imDims[1],imDims[2])
        plot_image(w)
    if isinstance(figTitle, str): fig.suptitle(figTitle)


def plot_conv_weights(kernels, bias=None, numToPlot=20, figTitle=None):
    kernels=kernels.detach().to(dtype=torch.float32).to(torch.device("cpu"))
    numToPlot=np.minimum(numToPlot,kernels.size(0))
    if np.prod(kernels.size()[2:])==1:
        #plot 1x1 weights
        channels=np.minimum(32,kernels.size(1))
        plt.figure()
        #for idx in np.arange(numToPlot):
        #    plt.plot(np.arange(channels),kernels[idx,0:channels,0,0],label = str(idx))
        #plt.legend()
        plot_image(kernels[:,:,0,0])
        if isinstance(figTitle, str): plt.title(figTitle)
    else:
        #channelStart=kernels.size(1)-1 #plot weights in last channel
        channelStart=0 #plot weights of 1st channel
        if channelStart+3<=kernels.size(1):
            channelEnd=channelStart+3 #plot 3 channels as RGB image
        else:
            channelEnd=channelStart+1 #plot 1 channel as grey image
        numCols=10
        numRows=int(np.maximum(1,np.ceil(numToPlot/numCols)))
        fig = plt.figure(figsize=(numCols, numRows))
        for idx in np.arange(numToPlot):
            fig.add_subplot(numRows, numCols, idx+1)
            plot_image(kernels[idx,channelStart:channelEnd].squeeze())
            if figTitle==None:
                if bias==None:
                    plt.title(str(idx))
                else:
                    plt.title('{:.2f}'.format(bias[idx].item()))
        if isinstance(figTitle, str): fig.suptitle(figTitle)

def plot_adversarials(images, clippedAdvs, labels, predictedLabelsClean, predictedLabelsAdv):
    #find the indeces of those samples that were initially correctly classified, but are incorrectly labelled after the attack
    success=(predictedLabelsAdv!=labels)
    idxSuccess=np.nonzero((torch.logical_and(success==True, predictedLabelsClean==labels)))
    print("  number of images whose predicted label is changed after adversarial attack = {}".format(idxSuccess.numel()))
    toPlot=min(24,idxSuccess.numel())
    #plot adversarial images
    fig = plt.figure(figsize=(6,4))
    for idx in np.arange(toPlot):
       fig.add_subplot(4, 6, idx+1)
       plot_image(clippedAdvs[int(idxSuccess[idx]),:,:,:], 0)
       plt.title("{}".format(labels[int(idxSuccess[idx])]), pad=-10, loc='left')
       plt.title("-->",pad=-10)
       plt.title("{}".format(predictedLabelsAdv[int(idxSuccess[idx])]), pad=-10, loc='right')
    #plot corresponding clean images, for comparison
    fig = plt.figure(figsize=(6,4))
    for idx in np.arange(toPlot):
       fig.add_subplot(4, 6, idx+1)
       plot_image(images[int(idxSuccess[idx]),:,:,:], 0)
       plt.title("{}".format(labels[int(idxSuccess[idx])]), pad=-10, loc='left')
       #plt.title("clean", pad=-10, loc='right')
      
def plot_joint_confidence_histograms(acceptPred1, legendText1, acceptPred2, legendText2, actType='activation', fileName=None):
      #define appearance
      #plt.style.use('seaborn-v0_8-poster')
      fontSize=42
      color1='tab:blue'
      color2='tab:orange'
      #plt.rcParams['font.size']=30
      #plt.rcParams.update({'font.size': 50})
      fig, ax1 = plt.subplots()
      ax2 = ax1.twinx()

      #define how to bin data
      acceptPred1=np.array(acceptPred1.to(torch.device("cpu")))
      acceptPred2=np.array(acceptPred2.to(torch.device("cpu")))
      #print(acceptPred1.min(),acceptPred2.min(),acceptPred1.max(),acceptPred2.max())
      binMax=acceptPred1.max()
      binMin=acceptPred1.min()
      if len(acceptPred2)>0: 
          binMin=np.minimum(binMin,acceptPred2.min())
      binMin=np.minimum(binMin,binMax-0.1)
      binMargin=0.05*(binMax-binMin)
      binMin=binMin-binMargin
      binMax=binMax+binMargin
      if binMin<0.1*binMax and binMin>0:
          binMin=0
      bins = np.arange(binMin,binMax,0.01*(binMax-binMin))

      #plot histograms
      #plt.hist([acceptPred1,acceptPred2], bins, label=[legendText1, legendText2])
      ax2.hist(acceptPred1, bins, label=legendText1, alpha=0.5, color=color2, rwidth=1)
      ax1.hist(acceptPred2, bins, label=legendText2, alpha=0.5, color=color1, rwidth=1)
      #fig.legend(loc='outside upper right', bbox_to_anchor=(0.925, 1.035), ncols=2, fancybox=True, fontsize = fontSize)

      #set appearance of axes
      ax1.set_xlabel(actType, fontsize = fontSize)
      ax2.set_ylabel(legendText1, fontsize = fontSize, color=color2, backgroundcolor='white', y=1, labelpad=0, ha='right', va='top', rotation="horizontal")
      ax1.set_ylabel(legendText2, fontsize = fontSize, color=color1, backgroundcolor='white', y=1, labelpad=0, ha='left', va='top', rotation="horizontal")
      ax2.locator_params(axis='both', nbins=5)   
      ax1.locator_params(axis='both', nbins=5)   
      ax2.tick_params(labelsize=0.9*fontSize)
      ax1.tick_params(labelsize=0.9*fontSize)
      ax2.tick_params(axis='y', colors=color2)
      ax1.tick_params(axis='y', colors=color1)
      ax1.set_ylim(top=ax1.get_ylim()[1]*1.1)
      if fileName is not None:
          plt.savefig(fileName+'_'+actType+'_hist_vs'+legendText2+'.pdf',format='pdf',bbox_inches='tight')
      plt.show()

def parse_string(string,base,divider='-',limit=None):
    #If divider='-', then function takes
    #a string of the form 'baseXX' and returns the list ['XX'], or for 
    #a string of the form 'baseXX-YY-ZZ' returns a list ['XX', 'YY', 'ZZ']
    #ignores anything after the limit
    if limit!=None:
        string=string.split(limit)[0]
    if len(base)>0:
        string=string.split(base)
        if np.size(string)<2:
            print('parse_string was applied to',string,'but did not find the initial string',base)
            string=string[0]
        else:
            string=string[1]
    params=string.split(divider)
    
    return params

def sign_preserving_power(x, exponent=2.0):
    y=torch.sign(x)*torch.pow(F.relu(torch.abs(x)),exponent)
    return y

def measure_sparsity_hoyer(y, dim=None, keepdim=False):
    if dim==None:
        n=y.numel()
    elif isinstance(dim, int):
        n=y.size(dim)
    else:
        n=1
        for d in dim: 
            n*=y.size(d)
    sqrtn=n**0.5
    norm1=torch.norm(y,p=1,dim=dim,keepdim=keepdim)+1e-9
    norm2=torch.norm(y,p=2,dim=dim,keepdim=keepdim)+1e-9
    sparsity=(sqrtn-(norm1/norm2))/(sqrtn-1)
    #sparsity=torch.where(sparsity.isnan(),torch.zeros_like(sparsity),sparsity)
    return sparsity

def delete_elements(x, ind):
    if torch.is_tensor(x):
        mask=torch.ones(x.size(0), dtype=torch.bool)
        mask[ind]=False
        x=x[mask]
    elif type(x) is list:
        ind=list(ind)
        for idx in sorted(ind, reverse = True): del x[idx]
    else:
        mask=np.ones(np.size(x,0), dtype=bool)
        ind=np.array(ind)
        mask[ind]=False
        x=x[mask]
    return x

def find_indices(x):
    #x is a boolean vector (such as might be produced by ind=a==b)
    indices=[]
    for idx in range(len(x)):
        if x[idx]:
            indices.append(idx)
    return indices

def chunked_indices(fullRange, chunkSize):
    # divides a range ("fullRange") up into non-overlapping sub-ranges of length "chunkSize",
    # last sub-range will be smaller than the others if the range is not exactly divisible
    # by the chunkSize
    # adapted from: https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks/74120449#74120449
    for i, c in enumerate(fullRange[::chunkSize]):
        yield fullRange[i*chunkSize:(i+1)*chunkSize]

def find_integer_factors(intVal, maxVal=None):
    # find all the fators of an integer value "intVal".
    # if "maxVal is defined, only values less than or equal to this are returned
    if maxVal==None: maxVal=intVal
    factors = set()
    for factor in range(1, 1+int(intVal**0.5)):
        if intVal % factor == 0:  # remainder is zero
            fTwin=intVal // factor
            if factor<=maxVal: factors.add(factor)
            if fTwin<=maxVal: factors.add(fTwin)
    return factors


def mean_non_zero(y, dim=None, keepdim=False, zeroEq=1e-6):
    #find the mean, but exclude values that equal zero from the calculation
    return torch.sum(y, dim=dim, keepdim=keepdim)/(zeroEq+torch.sum(torch.abs(y.detach())>0, dim=dim, keepdim=keepdim))

def mean_above_zero(y, dim=None, keepdim=False, zeroEq=1e-6):
    #find the mean, but exclude values that <= zero from the calculation
    y=F.relu(y)
    return torch.sum(y, dim=dim, keepdim=keepdim)/(zeroEq+torch.sum(y.detach()>0, dim=dim, keepdim=keepdim))

def mean_above_average(y, dim=None, keepdim=False, zeroEq=1e-6):
    #replace below average values with zeros
    thres=torch.mean(y.detach(), dim=dim, keepdim=True)
    y=torch.where(y>=thres.expand(y.size()),y,torch.zeros_like(y))
    return mean_above_zero(y, dim=dim, keepdim=keepdim, zeroEq=zeroEq)

def min_above_zero(y, dim=None, keepdim=False):
    #find the minimum value of y that is greater than zero
    #replace values that are <=0 with Inf:
    y=F.relu(y)
    tmp=torch.where(y==0,torch.inf*torch.ones_like(y),y)
    #find the minimum values:
    minVals=min_values(tmp, dim=dim, keepdim=keepdim)
    return minVals

def minimum_magnitude(a, b):
    # return the elements from a and b that are closest to zero
    idx=a.abs()<b.abs()
    c=b
    c[idx]=a[idx]
    return c

def boltzmann_magnitude(a, b, alpha=0):
    # combines corresponding values in two tensors using the boltzmann operator
    # alpha=0 produces the average
    # alpha->inf approximates the max value (a smooth approximation to torch.maximum(a,b))
    # alpha->-inf approximates the min value (a smooth approximation to torch.minimum(a,b))
    expa = torch.exp(alpha*a)
    expb = torch.exp(alpha*b)
    return (a*expa+b*expb)/(expa+expb)

def boltzmann_mag(y, alpha=0, dim=None, keepdim=False):
    # combines values in a tensor using the boltzmann operator
    # alpha=0 produces the average
    # alpha->inf approximates the max value (a smooth approximation to torch.max(y))
    # alpha->-inf approximates the min value (a smooth approximation to torch.min(y))

    # shift values to a range that will minimize numerical under/overflows (subtracting
    # a constant does not effect results)
    #shift=torch.mean(y, dim=dim, keepdim=True)
    shift = (max_values(y, dim=dim, keepdim=True)+min_values(y, dim=dim, keepdim=True))/2
    y = y - shift

    # apply the boltzmann operator
    yAlphaExp = torch.exp(alpha * y)
    z = torch.sum(y * yAlphaExp, dim=dim, keepdim=keepdim)
    z = z / torch.sum(yAlphaExp, dim=dim, keepdim=keepdim)
    return z

def min_values(y, dim=None, keepdim=False):
    #a wrapper for torch's min function that always returns the minimum values 
    #(not the indeces) and allows a tuple of multiple dimensions to be specified
    if dim==None:
        z=torch.min(y)
    elif isinstance(dim, int):
        z=torch.min(y,dim=dim,keepdim=keepdim).values
    else:
        dim=torch.sort(torch.tensor(dim), descending=True).values
        z=y
        for idx in dim:
            z=torch.min(z,dim=idx,keepdim=keepdim).values
    return z

def max_values(y, dim=None, keepdim=False):
    #a wrapper for torch's max function that always returns the maximum values 
    #(not the indeces) and allows a tuple of multiple dimensions to be specified
    if dim==None:
        z=torch.max(y)
    elif isinstance(dim, int):
        z=torch.max(y,dim=dim,keepdim=keepdim).values
    else:
        dim=torch.sort(torch.tensor(dim), descending=True).values
        z=y
        for idx in dim:
            z=torch.max(z,dim=idx,keepdim=keepdim).values
    return z


def off_diagonal(x):
    #from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def gauss2d(sigma=1.5, orient=0, aspect=1, norm=1, pxsize=None, cntr=[0,0], order=0):
    """    
    Produces a numerical approximation to Gaussian (-derivative) function with
    variable aspect ratio.
    Parameters:
    sigma  = standard deviation of Gaussian envelope
    orient = orientation of the Gaussian clockwise from the vertical (degrees)
    aspect = aspect ratio of Gaussian envelope (0 = no "length" to envelope, 
             1 = circular symmetric envelope)
    norm   = 1 to normalise the gaussian so that it sums to 1
           = 0 for no normalisation (gaussian has max value of 1)
    pxsize = the size of the filter.
             Optional, if not specified size is 6*sigma*max(1,aspect)
    cntr   = location of the centre of the gaussian.
             Optional, if not specified gaussian is centred in the middle of image.
    order  = order of differential. Differential is calculated in the y direction.
             Valid values are whole numbers from 0 (output is not differentiated) 
             to 4 (output is the 4th differential of a gaussian).
    """
    #avoid division by zero errors
    sigma=np.maximum(1e-15,sigma);

    if pxsize==None:
        pxsize=np.ones([2])*sigma*6*np.maximum(1,aspect)

    #define grid of x,y coodinates at which to define function
    x = np.linspace(-pxsize[0]/2, pxsize[0]/2, int(pxsize[0]))
    y = np.linspace(-pxsize[1]/2, pxsize[1]/2, int(pxsize[1]))
    x, y=np.meshgrid(x,y)
     
    #rotate 
    orient=-orient*np.pi/180
    x_theta=(x-cntr[0])*np.cos(orient)+(y-cntr[1])*np.sin(orient)
    y_theta=-(x-cntr[0])*np.sin(orient)+(y-cntr[1])*np.cos(orient)
    
    
    #define gaussian
    gauss=np.exp(-0.5*( ((x_theta**2)/((sigma*aspect)**2)) + ((y_theta**2)/(sigma**2)) ))
    gauss=gauss/(sigma**2*(2*np.pi))
    
    #differentiate gaussian, if requested
    if order==1:
      gauss=gauss*(-y_theta/(sigma**2))
    elif order==2:
      gauss=gauss*(((y_theta**2)-(sigma**2))/(sigma**4))
    elif order==3:
      gauss=gauss*(-y_theta*((y_theta**2)-(3*sigma**2))/(sigma**6))
    elif order==4:
      gauss=gauss*(((y_theta**4)-(6.*y_theta**2*sigma**2)+(3*sigma**4))/(sigma**8))
    elif order>4:
      print('ERROR: order of differential applied to gaussian, not defined')
      
    #normalise
    if norm:
        gauss=gauss/np.sum(np.abs(gauss))
    else:
        gauss=gauss/np.max(np.abs(gauss))

    return gauss


def tensor_to_image(imTensor):
    #convert a pytorch tensor (B,C,H,W) to a numpy image (H,W,C)
    #note: discards all but 1st item in batch
    imTensor=imTensor[0]
    image=torch.transpose(imTensor,0,2)
    return image

def image_to_tensor(image):
    #convert a numpy image (H,W,C) to a pytorch tensor (B,C,H,W)
    imTensor=torch.from_numpy(image).type(torch.FloatTensor)
    imTensor=torch.transpose(imTensor,0,2)
    imTensor=imTensor[None,:,:,:]
    return imTensor


def print_error(text):
    print('\033[31m' + text + '\033[0m')

def print_warning(text):
    print('\x1b[1;33m' + text + '\033[0m')

#import psutil
def available_ram(device, units='bytes'):
    # determines the memory that is available for use on a device
    scale = 1
    if units.startswith('k'): # give result in kilobytes
        scale = 1024
    elif units.startswith('M'): # give result in megabytes
        scale = 1024**2
    elif units.startswith('G'): # give result in gigabytes
        scale = 1024**3

    if device == torch.device("cpu"):
        freeBytes = psutil.virtual_memory().available
    else:
        freeBytes = torch.cuda.memory_reserved(device)-torch.cuda.memory_allocated(device)
        #freeBytes, _ = torch.cuda.mem_get_info(device)
    return freeBytes / scale
