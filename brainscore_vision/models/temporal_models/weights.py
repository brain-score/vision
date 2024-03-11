import os

HOME_DIR = "/home/ytang/workspace/data/weights"

class WeightRegistry(dict):
    def __getitem__(self, model_weight_path):
        return os.path.join(HOME_DIR, model_weight_path)
    

weight_registry = WeightRegistry()