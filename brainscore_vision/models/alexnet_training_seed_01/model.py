import torch
import torch.nn as nn
import numpy as np
from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

def alexnet_v2_pytorch(num_classes=1000, dropout_keep_prob=0.5, global_pool=False):
    """
    Instantiate the AlexNetV2 model in PyTorch.
    """
    class AlexNetV2(nn.Module):
        def __init__(self, num_classes, dropout_keep_prob, global_pool):
            super(AlexNetV2, self).__init__()
            self.global_pool = global_pool

            # Convolutional layers
            self.features = nn.Sequential(
                #conv1
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                #conv2
                nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                #conv3
                nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                #conv4
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),

                #conv5
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )

            # Fully connected layers
            self.classifier = nn.Sequential(
                #fc6
                nn.Conv2d(256, 4096, kernel_size=5, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=1 - dropout_keep_prob),

                #fc7
                nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=1 - dropout_keep_prob),

                #fc8
                nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0)
            )

        def forward(self, x):
            x = self.features(x)
            if self.global_pool:
                x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = self.classifier(x)
            x = torch.flatten(x, start_dim=1)
            return x



def get_model_list():
    return ['alexnet_training_seed_01']

def get_model(name):
    assert name == 'alexnet_training_seed_01'
    
    model = alexnet_v2_pytorch()
    # Load the pretrained weights
    weights_path = r'.\training_seed_01.pth'
    state_dict = torch.load(weights_path)  # Load the .pth file
    
    # Extract the actual model weights
    model_weights = state_dict['model_state_dict']
    model.load_state_dict(model_weights)
    
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = PytorchWrapper(identifier='alexnet_training_seed_01', model=model, preprocessing=preprocessing)
    return ModelCommitment(identifier='alexnet_training_seed_01', activations_model=wrapper,
                            # specify layers to consider
                            layers=[
                                'features.0',  # conv1
                                'features.3',  # conv2
                                'features.6',  # conv3
                                'features.8',  # conv4
                                'features.10',  # conv5
                                'classifier.0',  # fc6
                                'classifier.3',  # fc7
                            ])
                            #fc8 is a classifier layer and is not included


def get_layers(name):
    assert name == 'alexnet_training_seed_01'
    return ['features.0',  # conv1
        'features.3',  # conv2
        'features.6',  # conv3
        'features.8',  # conv4
        'features.10',  # conv5
        'classifier.0',  # fc6
        'classifier.3',  # fc7
           ]

def get_bibtex(model_identifier):
    return """  
                Model introduced in:
                @misc{Mehrer_2020,
                title={Individual differences among deep neural network models},
                url={osf.io/3xupm},
                DOI={10.17605/OSF.IO/3XUPM},
                publisher={OSF},
                author={Mehrer, Johannes},
                year={2020},
                month={Oct}
}"""

if __name__ == '__main__':
    check_models.check_base_models(__name__)
