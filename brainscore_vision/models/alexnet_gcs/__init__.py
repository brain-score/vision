from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_gcs_model, LAYERS, GCS_LAYERS

# model_registry['alexnet'] = lambda: ModelCommitment(
#     identifier='alexnet',
#     activations_model=get_model(),
#     layers=LAYERS)

# for fov in [5, 10, 20, 25]:
model_registry['alexnet_gcs_FOV-5'] = lambda: ModelCommitment(
    identifier='alexnet_gcs_FOV-5',
    activations_model=get_gcs_model(fov=5),
    layers=GCS_LAYERS)
model_registry['alexnet_gcs_FOV-10'] = lambda: ModelCommitment(
    identifier='alexnet_gcs_FOV-10',
    activations_model=get_gcs_model(fov=10),
    layers=GCS_LAYERS)
model_registry['alexnet_gcs_FOV-20'] = lambda: ModelCommitment(
    identifier='alexnet_gcs_FOV-20',
    activations_model=get_gcs_model(fov=20),
    layers=GCS_LAYERS)
# model_registry['alexnet_gcs_FOV-25'] = lambda: ModelCommitment(
#     identifier='alexnet_gcs_FOV-25',
#     activations_model=get_gcs_model(fov=25),
#     layers=GCS_LAYERS)
