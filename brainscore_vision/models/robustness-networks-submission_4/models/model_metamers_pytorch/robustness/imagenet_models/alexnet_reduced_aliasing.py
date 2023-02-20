import torch.nn as nn
try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torch.hub import load_state_dict_from_url

from .custom_modules import FakeReLUM, ConvHPool2d, HannPooling2d

__all__ = ['AlexNet_reduced_aliasing', 'alexnet_reduced_aliasing']

model_urls = {
}


class AlexNet_reduced_aliasing(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet_reduced_aliasing, self).__init__()
        self.features = nn.Sequential(
            # Original network had padding=2 for kernel_size=11 (lose 3 pixels from each side)
            # With pool_size=17 padding=5 
            ConvHPool2d(3, 64, kernel_size=11, pool_size=17, stride=4, padding=5),
            nn.ReLU(inplace=False),
            HannPooling2d(stride=2, pool_size=9, padding=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            HannPooling2d(stride=2, pool_size=9, padding=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            HannPooling2d(stride=2, pool_size=9, padding=2),
        )
        featurenames = ['conv0', 'relu0', 'hpool0',
                        'conv1', 'relu1', 'hpool1',
                        'conv2', 'relu2',
                        'conv3', 'relu3',
                        'conv4', 'relu4',
                        'hpool2']
        self.featurenames = featurenames

        self.fake_relu_dict = nn.ModuleDict()
        for layer_name in self.featurenames:
            if 'relu' in layer_name:
                self.fake_relu_dict[layer_name] =  FakeReLUM()

        # If the input shape is the original alexnet shape of 224x224 this 
        # operation does nothing. 
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes)    
        )
        self.classifier_names = ['dropout0', 'fc0', 'fc0_relu',
                                 'dropout1', 'fc1', 'fc1_relu',
                                 'fctop']
        self.fake_relu_dict['fc0_relu'] = FakeReLUM()
        self.fake_relu_dict['fc1_relu'] = FakeReLUM()

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        all_outputs = {}
        all_outputs['input_after_preproc'] = x

        for layer, name in list(zip(self.features, self.featurenames)):
            if ('relu' in name) and fake_relu:
                all_outputs[name + '_fake_relu'] = self.fake_relu_dict[name](x)
            x = layer(x)
            all_outputs[name] = x

        x = self.avgpool(x)
        all_outputs['avgpool'] = x

        x = x.view(x.size(0), 256 * 6 * 6)
        all_outputs['xview'] = x

        for layer, name in list(zip(self.classifier, self.classifier_names)):
            if ('relu' in name) and fake_relu:
                all_outputs[name + '_fake_relu'] = self.fake_relu_dict[name](x)
            x = layer(x)
            all_outputs[name] = x

        all_outputs['final'] = all_outputs['fctop']

        if with_latent and no_relu:
            raise ValueError('no_relu is deprecated') 
            return x, None, all_outputs
        if with_latent:
            return x, None, all_outputs
        return x

def alexnet_reduced_aliasing(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper. 
    Modified for reduced aliasing using hanning pooling

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet_reduced_aliasing(**kwargs)
    if pretrained:
        raise ValueError('No pretrained model exists for alexnet reduced aliasing')
    return model
