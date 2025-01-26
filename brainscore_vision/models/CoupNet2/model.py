import os
import torch
from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from .singleccnn import *
import torch.nn as nn


# This is an example implementation for submitting resnet-50 as a pytorch model
# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.


def get_model(name):
  assert name == 'CoupNet2'
  model = load_model(pretrained_model)
  preprocessing = functools.partial(load_preprocess_images, image_size=28)
  wrapper = PytorchWrapper(identifier='CoupNet2', model=model, preprocessing=preprocessing)
  wrapper.image_size = 28
  return wrapper


def get_layers(name):
  assert name == 'CoupNet2'
  return ['conv1','conv2', 'fc1', 'fc2']


def get_bibtex(model_identifier):
  return """"""


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = CoupNet2(6)  # time step is set to 6
    model.load_state_dict(checkpoint['model'])
    return model

# 预训练模型路径(训练好的模型文件的存储路径)
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 构建 checkpoint 文件的绝对路径
pretrained_model = os.path.join(current_script_path, 'mnist_checkpoint_max.pth')

class SeqToANNContainer(nn.Module):
    # 摘自SpikingJelly。将时间维度上的操作转换为并行。[N,T,C,W,H] -> [NT,C,W,H] -> [N,T,C,W,H]
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]  # 获取第一维度和第二维度的大小
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())   # 第一、二维度合并后的张量进行传入的参数的操作
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class CoupNet2(nn.Module):
    def __init__(self, t):
        super(CoupNet2, self).__init__()
        self.T = t

        self.conv1 = nn.Sequential(SeqToANNContainer(nn.Conv2d(1, 128, 3, 1, 1, bias=False)),
                                   CCNN2d(self.T, 128, settled_weight=False), )
        self.conv2 = nn.Sequential(SeqToANNContainer(nn.Conv2d(128, 128, 3, 1, 1, bias=False)),
                                   CCNN2d(self.T, 128, settled_weight=False), )
        self.pool = SeqToANNContainer(nn.MaxPool2d(2))

        self.fc1 = nn.Sequential(SeqToANNContainer(nn.Linear(128 * 14 * 14, 128)),
                                 CCNN1d(self.T, 128, settled_weight=False), )
        self.fc2 = nn.Sequential(SeqToANNContainer(nn.Linear(128, 10)),
                                 CCNN1d(self.T, 10, settled_weight=False), )

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=2)
        x = self.fc1(x)
        x = self.fc2(x)
        return x.mean(dim=1)


  if __name__ == '__main__':
      check_models.check_base_models(__name__)