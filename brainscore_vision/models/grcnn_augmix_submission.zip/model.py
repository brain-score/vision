## Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from model_tools.check_submission import check_models
import numpy as np
import torch
#from torch import nn
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model
#from candidate_models import s3 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functools import reduce
import math
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


   
import logging
import os
import sys
### S3 download 
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

_logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = 'brainscore2022'
_DEFAULT_REGION = 'us-east-1'
_NO_SIGNATURE = Config(signature_version=UNSIGNED)


def download_folder(folder_key, target_directory, bucket=_DEFAULT_BUCKET, region=_DEFAULT_REGION):
    if not folder_key.endswith('/'):
        folder_key = folder_key + '/'
    s3 = boto3.resource('s3', region_name=region, config=_NO_SIGNATURE)
    bucket = s3.Bucket(bucket)
    bucket_contents = list(bucket.objects.all())
    files = [obj.key for obj in bucket_contents if obj.key.startswith(folder_key)]
    _logger.debug(f"Found {len(files)} files")
    for file in tqdm(files):
        # get part of file after given folder_key
        filename = file[len(folder_key):]
        if len(filename) > 0:
            target_path = os.path.join(target_directory, filename)
            temp_path = target_path + '.filepart'
            bucket.download_file(file, temp_path)
            os.rename(temp_path, target_path)


def download_file(key, target_path, bucket=_DEFAULT_BUCKET, region=_DEFAULT_REGION):
    s3 = boto3.resource('s3', region_name=region, config=_NO_SIGNATURE)
    obj = s3.Object(bucket, key)
    # show progress. see https://gist.github.com/wy193777/e7607d12fad13459e8992d4f69b53586
    with tqdm(total=obj.content_length, unit='B', unit_scale=True, desc=key, file=sys.stdout) as progress_bar:
        def progress_hook(bytes_amount):
            progress_bar.update(bytes_amount)

        obj.download_file(target_path, Callback=progress_hook)




"""
Template module for a base model submission to brain-score
"""






# define your custom model here:
class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32, groups=32):

        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList() 
        for i in range(M):

            conv1 = nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=groups,bias=False)
            init.kaiming_normal_(conv1.weight)
            self.conv.append(nn.Sequential(conv1,
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1) 
        conv_fc = nn.Conv2d(out_channels,d,1,bias=False)
        init.normal_(conv_fc.weight, std=0.01)
        self.fc1=nn.Sequential(conv_fc,
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)
        init.normal_(self.fc2.weight, std=0.01)
        self.softmax=nn.Softmax(dim=1) 

    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        for i,conv in enumerate(self.conv):
            output.append(conv(input))
        U=reduce(lambda x,y:x+y,output) 
        s=self.global_pool(U)
        z=self.fc1(s)  
        a_b=self.fc2(z) 
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1)
        a_b=self.softmax(a_b) 
        a_b=list(a_b.chunk(self.M,dim=1))
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) 
        V=list(map(lambda x,y:x*y,output,a_b)) 
        V=reduce(lambda x,y:x+y,V) 
        return V

class GRCL(nn.Module):
  def __init__(self, inplanes, planes, downsample=True, iter = 3, SKconv=True, expansion=2):
    super(GRCL, self).__init__()

    self.iter = iter
    self.expansion = expansion
    # feed-forward part
    self.add_module('bn_f', nn.BatchNorm2d(inplanes))
    self.add_module('relu_f', nn.ReLU(inplace=True))
    conv_f = nn.Conv2d(inplanes, int(planes* self.expansion), kernel_size=3, stride=1, padding=1, bias=False, groups=32)
    init.kaiming_normal_(conv_f.weight)
    self.add_module('conv_f', conv_f)
    
    self.add_module('bn_g_f', nn.BatchNorm2d(inplanes))
    self.add_module('relu_g_f', nn.ReLU(inplace=True))
    conv_g_f = nn.Conv2d(inplanes, int(planes* self.expansion), kernel_size=1, stride=1, padding=0, bias=True, groups=32)
    init.normal_(conv_g_f.weight, std=0.01)
    self.add_module('conv_g_f', conv_g_f)
    self.conv_g_r = nn.Conv2d(int(planes* self.expansion), int(planes* self.expansion), kernel_size=1, stride=1, padding=0, bias=False, groups=32)
    self.add_module('sig', nn.Sigmoid())

    # recurrent part
    for i in range(0, self.iter):
     layers = []
     layers_g_bn = []
    
     layers.append(nn.BatchNorm2d(planes*self.expansion))
     layers.append(nn.ReLU(inplace=True))
     conv_1 = nn.Conv2d(int(planes*self.expansion), planes, kernel_size=1, stride=1, padding=0, bias=False)
     init.kaiming_normal_(conv_1.weight)
     layers.append(conv_1)

     layers.append(nn.BatchNorm2d(planes))
     layers.append(nn.ReLU(inplace=True))

     if SKconv:
       layers.append(SKConv(planes, planes))
     else:
       layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
       layers.append(nn.BatchNorm2d(planes))
       layers.append(nn.ReLU(inplace=True))

     conv_2 = nn.Conv2d(planes, int(planes*self.expansion), kernel_size=1, stride=1, padding=0, bias=False)   
     init.kaiming_normal_(conv_2.weight)
     layers.append(conv_2)
     layers_g_bn.append(nn.BatchNorm2d(int(planes*self.expansion)))

     layers_g_bn.append(nn.ReLU(inplace=True)) 

     self.add_module('iter_'+str(i+1), nn.Sequential(*layers))
     self.add_module('iter_g_'+str(i+1), nn.Sequential(*layers_g_bn))

    self.downsample = downsample
    if self.downsample:
       self.add_module('d_bn', nn.BatchNorm2d(planes * self.expansion))
       self.add_module('d_relu', nn.ReLU(inplace=True))
       d_conv = nn.Conv2d(int(planes* self.expansion), int(planes* self.expansion), kernel_size=1, stride=1, padding=0, bias=False)
       init.kaiming_normal_(d_conv.weight)
       self.add_module('d_conv', d_conv)
       self.add_module('d_ave', nn.AvgPool2d((2, 2), stride=2))
  
       self.add_module('d_bn_1', nn.BatchNorm2d(planes * self.expansion))
       self.add_module('d_relu_1', nn.ReLU(inplace=True))
       d_conv_1 = nn.Conv2d(int(planes* self.expansion), planes, kernel_size=1, stride=1, padding=0,
       bias=False)
       init.kaiming_normal_(d_conv_1.weight)
       self.add_module('d_conv_1', d_conv_1)

       self.add_module('d_bn_3', nn.BatchNorm2d(planes))
       self.add_module('d_relu_3', nn.ReLU(inplace=True))
       
       if SKconv:
         d_conv_3 = SKConv(planes, planes, stride=2)
         self.add_module('d_conv_3', d_conv_3)
       else:
         d_conv_3 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False)
         init.kaiming_normal_(d_conv_3.weight)
         self.add_module('d_conv_3', d_conv_3)

       d_conv_1e = nn.Conv2d(planes, int(planes * self.expansion), kernel_size=1, stride=1, padding=0, bias=False)
       init.kaiming_normal_(d_conv_1e.weight)
       self.add_module('d_conv_1e', d_conv_1e)

  def forward(self, x):
    # feed-forward
    x_bn = self.bn_f(x)
    x_act = self.relu_f(x_bn)
    x_s = self.conv_f(x_act)
    
    x_g_bn = self.bn_g_f(x)
    x_g_act = self.relu_g_f(x_g_bn)
    x_g_s = self.conv_g_f(x_g_act)

    # recurrent 
    for i in range(0, self.iter):
       x_g_r = self.conv_g_r(self.__dict__['_modules']["iter_g_%s" % str(i+1)](x_s))
       x_s = self.__dict__['_modules']["iter_%s" % str(i+1)](x_s) * torch.sigmoid(x_g_r + x_g_s) + x_s

    if self.downsample:
      x_s_1 = self.d_conv(self.d_ave(self.d_relu(self.d_bn(x_s))))
      x_s_2 = self.d_conv_1e(self.d_conv_3(self.d_relu_3(self.d_bn_3(self.d_conv_1(self.d_relu_1(self.d_bn_1(x_s)))))))
      x_s = x_s_1 + x_s_2

    return x_s

class GRCNN(nn.Module):
 
  def __init__(self, iters, maps, SKconv, expansion, num_classes):
    """ Args:
      iters:iterations.
      num_classes: number of classes
    """
    super(GRCNN, self).__init__()
    self.iters = iters
    self.maps = maps
    self.num_classes = num_classes
    self.expansion = expansion

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

    init.kaiming_normal_(self.conv1.weight)

    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)

    init.kaiming_normal_(self.conv2.weight)    

    self.layer1 = GRCL(64, self.maps[0], True, self.iters[0], SKconv, self.expansion)
    self.layer2 = GRCL(self.maps[0] * self.expansion, self.maps[1], True, self.iters[1], SKconv, self.expansion)
    self.layer3 = GRCL(self.maps[1] * self.expansion, self.maps[2], True, self.iters[2], SKconv, self.expansion)
    self.layer4 = GRCL(self.maps[2] * self.expansion, self.maps[3], False, self.iters[3], SKconv, self.expansion)

    self.lastact = nn.Sequential(nn.BatchNorm2d(self.maps[3]*self.expansion), nn.ReLU(inplace=True))
    self.avgpool = nn.AvgPool2d(7)
    self.classifier = nn.Linear(self.maps[3] * self.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if m.bias is not None:
           init.zeros_(m.bias)
      elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)
      elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        init.zeros_(m.bias)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.conv2(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.lastact(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)

def grcnn55(num_classes=1000):
  """
  Args:
    num_classes (uint): number of classes
  """
  model = GRCNN([3, 3, 4, 3], [64, 128, 256, 512], SKconv=False, expansion=4, num_classes=num_classes)
  return model



# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_images, image_size=224)
#dir_path = os.path.dirname(os.path.realpath(""))
download_file("model_best.pth", "model_best.pth")

#model.load_state_dict(torch.load(dir_path + "/my_model.pth"))
model_ft = grcnn55() #models.resnet50(pretrained=True)
model_ft.load_state_dict(torch.load("model_best.pth", map_location = device)["state_dict"])
all_layers = [layer for layer, _ in model_ft.named_modules()]
all_layers = all_layers[1:]
all_layers = ['conv1', 'conv2', 'layer1', 'layer1.conv_g_f' , 'layer1.iter_1.8', 
              'layer1.iter_g_1.1', 'layer1.iter_2.8', 'layer1.iter_g_2.1', 'layer1.iter_3.3', 'layer1.iter_3.8' , 
              'layer1.iter_g_3.1', 'layer1.d_conv_1', 'layer1.d_conv_3', 'layer2.conv_g_r', 
              'layer2.iter_1.8', 'layer2.iter_g_1.1', 'layer2.iter_2.8' ,'layer2.iter_g_2.1', 'layer2.iter_3.8', 
              'layer2.iter_g_3.1', 'layer3.conv_f', 'layer3.conv_g_r', 'layer3.iter_1.8', 'layer3.iter_g_1.1' ,
              'layer3.iter_2.8', 'layer3.iter_g_2.1', 'layer3.iter_3.8', 'layer3.iter_g_3.1', 'layer3.iter_4.8', 
              'layer3.iter_g_4.1', 'layer3.d_conv_1e', 'layer4.conv_g_r', 'layer4.iter_1.8', 'layer4.iter_g_1.1', 
              'layer4.iter_2.8', 'layer4.iter_g_2.1', 'layer4.iter_3.8', 'layer4.iter_g_3.1', 
              'lastact.1',  'classifier']
#print(all_layers)
model_ft = model_ft.to(device)


# get an activations model from the Pytorch Wrapper
activations_model = PytorchWrapper(identifier='grcnn_robust_v1', model= model_ft , preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='gcrnn_robust_v1', activations_model=activations_model,
                        # specify layers to consider
                        layers=all_layers)


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ['grcnn_robust_v1']


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model(name):
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :param name: the name of the model to fetch
    :return: the model instance
    """
    assert name == 'grcnn_robust_v1'

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 224
    return wrapper


# get_layers method to tell the code what layers to consider. If you are submitting a custom
# model, then you will most likley need to change this method's return values.
def get_layers(name):
    """
    This method returns a list of string layer names to consider per model. The benchmarks maps brain regions to
    layers and uses this list as a set of possible layers. The lists doesn't have to contain all layers, the less the
    faster the benchmark process works. Additionally the given layers have to produce an activations vector of at least
    size 25! The layer names are delivered back to the model instance and have to be resolved in there. For a pytorch
    model, the layer name are for instance dot concatenated per module, e.g. "features.2".
    :param name: the name of the model, to return the layers for
    :return: a list of strings containing all layers, that should be considered as brain area.
    """

    # quick check to make sure the model is the correct one:
    assert name == 'grcnn_robust_v1'

    # returns the layers you want to consider
    return  all_layers

# Bibtex Method. For submitting a custom model, you can either put your own Bibtex if your
# model has been published, or leave the empty return value if there is no publication to refer to.
def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    # from pytorch.py:
    return ''

# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)