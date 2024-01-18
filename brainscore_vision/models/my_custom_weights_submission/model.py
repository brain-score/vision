# Custom Pytorch model from:
# https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb

from brainscore_vision.model_helpers.check_submission import check_models
import torch
from torch import nn
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch.nn.functional as F
from pathlib import Path
from brainscore_vision.model_helpers import download_weights
import os

BIBTEX = """"""
LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']


def load_model(modelname='resnet', resume=None, nclasses_fine=1000, nclasses_coarse=20):
    if modelname == 'resnet':
        model = MyModel(n_classes_fine=nclasses_fine, n_classes_coarse=nclasses_coarse)
    else:
        raise ValueError("Architechture {} not valid.".format(modelname))

    checkpoint_file = resume
    print("=> loading checkpoint '{}'".format(checkpoint_file))
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'saved-weights')):
        os.makedirs(os.path.join(os.path.dirname(__file__), 'saved-weights'))
    download_weights(
    bucket='brainscore-vision', 
    folder_path='models/my-custom-weights-submission',
    filename_version_sha=[('resnet_coarse_cifar100_b64_n161_160.pth', 'BuTXFYO48C__dXW9egQ6UQjj__m1T52d', '2360ff8352c500284feceee1c21c06c7b5081821')],
    save_directory=os.path.join(Path(__file__).parent,"saved-weights"))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)  # all key's don't have to match
    return model


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _ResBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(_ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class MyModel(nn.Module):
    def __init__(self, n_channels=3, block=_ResBlock, num_blocks=[2, 2, 2, 2], n_classes_fine=1000, n_classes_coarse=20):
        super(MyModel, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(n_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_coarse = nn.Linear(512 * block.expansion, n_classes_coarse)
        self.linear_fine_newName = nn.Linear(512 * block.expansion, n_classes_fine)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear_fine_newName(out)  # , self.linear_coarse(out)
        return out  # , hs


# get_model method actually gets the model. For a custom model, this is just linked to the
# model we defined above.
def get_model():
    """
    This method fetches an instance of a base model. The instance has to be callable and return a xarray object,
    containing activations. There exist standard wrapper implementations for common libraries, like pytorch and
    keras. Checkout the examples folder, to see more. For custom implementations check out the implementation of the
    wrappers.
    :return: the model instance
    """
    # init the model and the preprocessing:
    preprocessing = functools.partial(load_preprocess_images, image_size=32)
    dirname = os.path.dirname(__file__)
    weights_path = os.path.join(dirname, 'saved-weights/resnet_coarse_cifar100_b64_n161_160.pth')
    print(f"weights path is: {weights_path}")
    model = load_model(resume=weights_path)

    # get an activations model from the Pytorch Wrapper
    activations_model = PytorchWrapper(identifier='my-weights-model', model=model, preprocessing=preprocessing)

    # link the custom model to the wrapper object(activations_model above):
    wrapper = activations_model
    wrapper.image_size = 32
    return wrapper


# Main Method: In submitting a custom model, you should not have to mess with this.
if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    # dirname = os.path.dirname(__file__)
    # print(dirname)
    # print (os.getcwd())
    check_models.check_base_models(__name__)
