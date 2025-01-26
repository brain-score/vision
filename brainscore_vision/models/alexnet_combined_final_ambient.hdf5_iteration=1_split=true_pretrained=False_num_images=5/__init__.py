
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, get_layers
model_registry['alexnet_combined_final_ambient.hdf5_iteration\=1_split\=true_pretrained\=False_num_images\=5'] = lambda: ModelCommitment(identifier='alexnet_combined_final_ambient.hdf5_iteration\=1_split\=true_pretrained\=False_num_images\=5',
                                                                                                                                         activations_model=get_model('alexnet_combined_final_ambient.hdf5_iteration\=1_split\=true_pretrained\=False_num_images\=5'),
                                                                                                                                         layers=get_layers('alexnet_combined_final_ambient.hdf5_iteration\=1_split\=true_pretrained\=False_num_images\=5')))
