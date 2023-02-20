from .aacn import AACN_Model


class ResNet18AACN:

    def __init__(self, train_from_scratch=True, path=None):
        self.path = path
        self.train_from_scratch = train_from_scratch

    def get_model(self):
        # with aacn attention: attention=[False, True, True, True]
        model = AACN_Model.resnet18(num_classes=8, attention=[False, True, True, True], num_heads=4, k=0.25, v=0.25,
                                    image_size=224)

        return model
