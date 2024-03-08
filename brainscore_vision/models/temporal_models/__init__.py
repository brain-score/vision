from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.utils import get_specified_layers


def commit_model(identifier, module):
    activations_model=module.get_model(identifier)
    layers=get_specified_layers(activations_model)
    region_layer_map={"BRAIN": layers, "IT": [-2]}  # A hack here: we assume the second last layer is IT now, will provide a better way to specify this later
    return ModelCommitment(identifier=identifier, activations_model=activations_model, layers=layers, region_layer_map=region_layer_map)

# torchvision
from .torchvision_models import base_models as torchvision_models

model_registry['r3d_18'] = lambda: commit_model('r3d_18', torchvision_models)
model_registry['r2plus1d_18'] = lambda: commit_model('r2plus1d_18', torchvision_models)
model_registry['mc3_18'] = lambda: commit_model('mc3_18', torchvision_models)
model_registry['s3d'] = lambda: commit_model('s3d', torchvision_models)
model_registry['mvit_v1_b'] = lambda: commit_model('mvit_v1_b', torchvision_models)
model_registry['mvit_v2_s'] = lambda: commit_model('mvit_v2_s', torchvision_models)


# mmaction2

from .mmaction2 import base_models as mmaction2_models
model_registry["I3D"] = lambda: commit_model("I3D", mmaction2_models)
model_registry["I3D-nonlocal"] = lambda: commit_model("I3D-nonlocal", mmaction2_models)
model_registry["SlowFast"] = lambda: commit_model("SlowFast", mmaction2_models)
model_registry["X3D"] = lambda: commit_model("X3D", mmaction2_models)
model_registry["TimeSformer"] = lambda: commit_model("TimeSformer", mmaction2_models)
model_registry["VideoSwin-B"] = lambda: commit_model("VideoSwin-B", mmaction2_models)
model_registry["VideoSwin-L"] = lambda: commit_model("VideoSwin-L", mmaction2_models)
model_registry["UniFormer-V1"] = lambda: commit_model("UniFormer-V1", mmaction2_models)
model_registry["UniFormer-V2-B"] = lambda: commit_model("UniFormer-V2-B", mmaction2_models)
model_registry["UniFormer-V2-L"] = lambda: commit_model("UniFormer-V2-L", mmaction2_models)