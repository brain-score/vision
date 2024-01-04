from model_tools.check_submission import check_models
import numpy as np
import torch
import argparse
from torch import nn
import functools
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
from model_tools.activations.pytorch import load_preprocess_images
from brainscore import score_model
import torchvision.models
import os

"""
Template module for a base model submission to brain-score
"""
from srf.structured_conv_layer import Srf_layer_shared
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='resnet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--CtF', type=eval, default=False, choices = [True, False])

parser.add_argument('--save', type=str, default='./experiment_'+__file__[:-3])
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args, unknown = parser.parse_known_args()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
#------------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class conv1(nn.Module):
    def __init__(self, in_channel, out_channel, out_shape=None):
        super(conv1, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.out_shape = out_shape

        self.conv = nn.Conv2d(self.in_channel,self.out_channel,3,1)
        self.output = Identity()

    def forward(self, x=None, state= None, batch_size=None):
        # if x is None:  # at t=0, there is no input yet except to V1
        #     x = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
        #     if self.conv.weight.is_cuda:
        #         x = x.cuda()

        # if state is None:
        #     state = 0

        # skip = x + state

        x = self.conv(x)
        state = self.output(x)
        output = state

        return output, state

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, out_shape = None):
        super(ResBlock, self).__init__()

        self.in_channels = inplanes
        self.out_channels = planes
        self.out_shape = out_shape

        self.norm1 = norm(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(self.in_channels , self.out_channels, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        self.output = Identity()

    def forward(self, x=None, state=None, batch_size=None):

        if x is None:  # at t=0, there is no input yet except to V1
            x = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
            if self.conv1.weight.is_cuda:
                x = x.cuda()

        shortcut = x

        if state is None:
            state = 0

        skip = x + state

        x = self.relu(self.norm1(skip))
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv2(x)


        state = self.output(x + shortcut)
        output = state

        return output, state


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([\
            transforms.RandomHorizontalFlip(),\
            transforms.RandomCrop(32, padding=4),\
            transforms.ToTensor(),\
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train), 
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.CIFAR10(root='./', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='./', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

class Downsampling(nn.Module):
    def __init__(self, in_channel, out_channel, out_shape):
        super(Downsampling, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.out_shape = out_shape

        self.DS_norm = norm(in_channel)
        self.DS_act  = nn.ReLU(inplace = True)
        self.DS_conv = nn.Conv2d(in_channel, out_channel, 4, 2, 1)
        self.output = Identity()

    def forward(self, x=None, state=None, batch_size=None):
        if x is None:  # at t=0, there is no input yet except to V1
            x = torch.zeros([batch_size, self.out_channel, self.out_shape, self.out_shape])
            if self.DS_conv.weight.is_cuda:
                x = x.cuda()
        else:
            x = self.DS_norm(x)
            x = self.DS_act(x)
            x = self.DS_conv(x)


        state = self.output(x)

        return x, state


class DCN(nn.Module):
    def __init__(self, times = 5):
        super(DCN, self).__init__()

        self.times = times

        # self.initialConv = nn.Conv2d(3,32,3,1)
        self.initialConv = conv1(3, 32, out_shape = 222)

        self.feature_layer1 = ResBlock(32, 32, out_shape = 222)

        self.DS1 = Downsampling(32, 64, out_shape = 111)

        self.feature_layer2 = ResBlock(64, 64, out_shape = 111)

        self.DS2 = Downsampling(64, 128, out_shape = 55)

        self.feature_layer3 = ResBlock(128, 128, out_shape = 55)

        self.fc_norm = norm(128)
        self.fc_act = nn.ReLU(inplace=True)
        self.fc_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_flatten = Flatten()
        self.fc_linear = nn.Linear(128,10)

    def forward(self, x):
        # x = self.initialConv(x)

        outputs = {'inp' : x}
        # states = {'inp' : 0}
        states = {}
        blocks = ['inp', 'initialConv', 'feature_layer1', 'DS1', 'feature_layer2', 'DS2', 'feature_layer3']

        for block in blocks[1:]:
            if block == 'initialConv':
                x = outputs['inp']
            else:
                x = None

            new_output, new_state = getattr(self, block)(x, batch_size=outputs['inp'].shape[0])

            # if block == 'feature_layer1':
            #     new_output, new_state = self.DS1(new_output)
            # elif block == 'feature_layer2':
            #     new_output, new_state = self.DS2(new_output)
            outputs[block] = new_output
            states[block] = new_state

        # print("States:",states)

        # print(outputs, states)
        sigma_vals = [5, 3, 1, 0.5, 0]
        original_input = outputs['inp']
        for t in range(1, self.times):
            # print(script_dir)
            # img = outputs['inp'][0] * 255
            # img = img.permute(1, 2, 0).detach().cpu().numpy()
            # cv.imwrite("image1.png", img)

            # blur = transforms.GaussianBlur(3, sigma=3.0)
            # img2 = blur(outputs['inp'][0]) * 255
            # img2 = img2.permute(1, 2, 0).detach().cpu().numpy()
            # cv.imwrite("image2.png", img2)
            # break
            if args.CtF == True:
                if sigma_vals[t] > 0:
                    blur = transforms.GaussianBlur(3, sigma= sigma_vals[t])
                    outputs['inp'] = blur(original_input)
                else:
                    outputs['inp'] = original_input

            # img = outputs['inp'][0] * 255
            # img = img.permute(1, 2, 0).detach().cpu().numpy()
            # cv.imwrite("img"+str(t)+".png", img)
            # print("IMAGE ", str(t))

            for block in blocks[1:]:

                prev_block = blocks[blocks.index(block) - 1]
                prev_output = outputs[prev_block]
                prev_state = states[block]


                new_output, new_state = getattr(self, block)(prev_output, prev_state)
                outputs[block] = new_output
                states[block] = new_state

        x = self.fc_norm(outputs['feature_layer3'])
        x = self.fc_act(x)
        x = self.fc_pool(x)
        x = self.fc_flatten(x)
        x = self.fc_linear(x)

        return x


# init the model and the preprocessing:
preprocessing = functools.partial(load_preprocess_images, image_size=224)

# get an activations model from the Pytorch Wrapper
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(torch.cuda.is_available(), device)

model = DCN()
# model = nn.DataParallel(model)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),'resnet_recur_adv.pth'), map_location = device)['state_dict'])
# model = model.module  #Remove DataParallel Wrapper
model.fc_linear = nn.Linear(128,1000) #Imagenet num_classes
print(model)
activations_model = PytorchWrapper(identifier='resnet_recur_adv_exp4_jun10', model=model, preprocessing=preprocessing)

# actually make the model, with the layers you want to see specified:
model = ModelCommitment(identifier='resnet_recur_adv_exp4_jun10', activations_model=activations_model,
                        # specify layers to consider
                        layers=['initialConv.conv', 'feature_layer1.conv2', 'DS1.DS_conv', 'feature_layer2.conv2', 'DS2.DS_conv', 'feature_layer3.conv2'])


# The model names to consider. If you are making a custom model, then you most likley want to change
# the return value of this function.
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ['resnet_recur_adv_exp4_jun10']


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
    assert name == 'resnet_recur_adv_exp4_jun10'

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
    assert name == 'resnet_recur_adv_exp4_jun10'

    # returns the layers you want to consider
    return ['initialConv.conv', 'feature_layer1.conv2', 'DS1.DS_conv', 'feature_layer2.conv2', 'DS2.DS_conv', 'feature_layer3.conv2']

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

