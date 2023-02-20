import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo
from .custom_modules import FakeReLUM
from .layers import conv2d_same, pool2d_same

__all__ = ['kell2018']


class AuditoryCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(AuditoryCNN, self).__init__()

        # Include initial batch norm to center the cochleagrams
        self.batchnorm0 = nn.BatchNorm2d(1)
       
        self.conv0 = conv2d_same.create_conv2d_pad(1, 96, kernel_size = [7, 14], stride = [3, 3], padding = 'same')
        self.relu0 = nn.ReLU(inplace = False)
        self.maxpool0 = pool2d_same.create_pool2d('max', kernel_size = [2,5] , stride = [2,2], padding = 'same')
        self.batchnorm1 = nn.BatchNorm2d(96)
        
        self.conv1 = conv2d_same.create_conv2d_pad(96, 256, kernel_size = [4,8], stride = [2,2], padding = 'same')
        self.relu1 = nn.ReLU(inplace = False)
        self.maxpool1 = pool2d_same.create_pool2d('max' , kernel_size = [2,5] , stride = [2,2], padding = 'same')
        self.batchnorm2 = nn.BatchNorm2d(256)
        
        self.conv2 = conv2d_same.create_conv2d_pad(256, 512, kernel_size = [2,5], stride = [1,1], padding = 'same')
        self.relu2 = nn.ReLU(inplace = False)
        
        self.conv3 = conv2d_same.create_conv2d_pad(512, 1024, kernel_size = [2,5], stride = [1,1], padding = 'same')
        self.relu3 = nn.ReLU(inplace = False)
        
        self.conv4 = conv2d_same.create_conv2d_pad(1024, 512, kernel_size = [2,5], stride = [1,1], padding = 'same')
        self.relu4 = nn.ReLU(inplace = False)
        
        self.avgpool = pool2d_same.create_pool2d('avg', kernel_size = [2,5] , stride = [2,2], padding = 'same')
        
        self.fullyconnected = nn.Linear( 512*9*5 , 4096)
        self.relufc = nn.ReLU(inplace = False)
        self.dropout = nn.Dropout()
        self.classification = nn.Linear(4096, num_classes)

        
        self.fake_relu_dict = nn.ModuleDict()
        self.fake_relu_dict['relu0'] =  FakeReLUM()
        self.fake_relu_dict['relu1'] =  FakeReLUM()
        self.fake_relu_dict['relu2'] =  FakeReLUM()
        self.fake_relu_dict['relu3'] =  FakeReLUM()
        self.fake_relu_dict['relu4'] =  FakeReLUM()
        self.fake_relu_dict['relufc'] =  FakeReLUM()
        

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        
        all_outputs = {}
        all_outputs['input_after_preproc'] = x
        
        x = self.batchnorm0(x)
        all_outputs['batchnorm0'] = x
        
        x = self.conv0(x)
        all_outputs['conv0'] = x
       
        if fake_relu:
            all_outputs['relu0' + '_fakerelu'] = self.fake_relu_dict['relu0'](x)
        
        x = self.relu0(x)
        all_outputs['relu0'] = x
        
        x = self.maxpool0(x)
        all_outputs['maxpool0'] = x
        
        x = self.batchnorm1(x)
        all_outputs['batchnorm1'] = x
        
        x = self.conv1(x)  
        all_outputs['conv1'] = x
   
        if fake_relu:
            all_outputs['relu1' + '_fakerelu'] = self.fake_relu_dict['relu1'](x)
        
        x = self.relu1(x)  
        all_outputs['relu1'] = x
        
        x = self.maxpool1(x) 
        all_outputs['maxpool1'] = x
        
        x = self.batchnorm2(x)  
        all_outputs['batchnorm2'] = x
        
        x = self.conv2(x) 
        all_outputs['conv2'] = x
   
        if fake_relu:
            all_outputs['relu2' + '_fakerelu'] = self.fake_relu_dict['relu2'](x)
        
        x = self.relu2(x)
        all_outputs['relu2'] = x
        
        x = self.conv3(x) 
        all_outputs['conv3'] = x
        
        if fake_relu:
            all_outputs['relu3' + '_fakerelu'] = self.fake_relu_dict['relu3'](x)
        
        x = self.relu3(x)
        all_outputs['relu3'] = x
        
        x = self.conv4(x) 
        all_outputs['conv4'] = x
   
        if fake_relu:
            all_outputs['relu4' + '_fakerelu'] = self.fake_relu_dict['relu4'](x)
        
        x = self.relu4(x)
        all_outputs['relu4'] = x
        
        x = self.avgpool(x)
        all_outputs['avgpool'] = x
        
        x = x.view(x.size(0), 512*9*5)
        all_outputs['xview'] = x
        
        x = self.fullyconnected(x)
        all_outputs['fullyconnected'] = x
        
        x_latent = x
   
        if fake_relu:
            all_outputs['relufc' + '_fakerelu'] = self.fake_relu_dict['relufc'](x)
        
        x_relu = self.relufc(x_latent)
        all_outputs['relufc'] = x_relu
 
        x_out = self.dropout(x_relu)
        all_outputs['dropout'] = x_out
        
        x_out = self.classification(x_out)
        all_outputs['final'] = x_out
        
        if with_latent and no_relu:
            return x_out, x_latent, all_outputs
        if with_latent:
            return x_out, x_relu, all_outputs
        return x_out
        
       
def kell2018(pretrained=False, progress=True, **kwargs):
    """KellNet model architecture from the Kell 2018. (Neuron)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    del pretrained # model checkpoint is not on model zoo
    model = AuditoryCNN(**kwargs)
        
    return model
