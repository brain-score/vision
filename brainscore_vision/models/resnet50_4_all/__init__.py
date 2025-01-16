
from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
import json
from pathlib import Path


import os
import importlib

# Dynamically import a module using __package__
def import_model():
    if __package__:
        module_name = f"{__package__}.model"
        return importlib.import_module(module_name)
    else:
        raise ImportError("This script is not part of a package and cannot use relative imports.")

# Use the dynamically imported module
model = import_model()


with open("config.json", "r") as f:
    MODEL_CONFIGS = json.load(f)

model_registry['resnet50_less_variation_iteration=1'] = lambda: ModelCommitment(identifier='resnet50_less_variation_iteration=1',
                                                                                 activations_model=model.get_model('resnet50_less_variation_iteration=1'), 
                                                                                 layers=MODEL_CONFIGS['resnet50_less_variation_iteration=1']['model_commitment']['layers'], 
                                                                                 region_layer_map=MODEL_CONFIGS['resnet50_less_variation_iteration=1']["region_layer_map"])

model_registry['resnet50_less_variation_iteration=2'] = lambda: ModelCommitment(identifier='resnet50_less_variation_iteration=2',
                                                                                activations_model=model.get_model('resnet50_less_variation_iteration=2'), 
                                                                                layers=MODEL_CONFIGS['resnet50_less_variation_iteration=2']['model_commitment']['layers'],
                                                                                region_layer_map=MODEL_CONFIGS['resnet50_less_variation_iteration=2']["region_layer_map"])

model_registry['resnet50_less_variation_iteration=3'] = lambda: ModelCommitment(identifier='resnet50_less_variation_iteration=3', 
                                                                                activations_model=model.get_model('resnet50_less_variation_iteration=3'), 
                                                                                layers=MODEL_CONFIGS['resnet50_less_variation_iteration=3']['model_commitment']['layers'],
                                                                                  region_layer_map=MODEL_CONFIGS['resnet50_less_variation_iteration=3']["region_layer_map"])

model_registry['resnet50_less_variation_iteration=4'] = lambda: ModelCommitment(identifier='resnet50_less_variation_iteration=4', 
                                                                                activations_model=model.get_model('resnet50_less_variation_iteration=4'),
                                                                                  layers=MODEL_CONFIGS['resnet50_less_variation_iteration=4']['model_commitment']['layers'], 
                                                                                  region_layer_map=MODEL_CONFIGS['resnet50_less_variation_iteration=4']["region_layer_map"])

model_registry['resnet50_less_variation_iteration=5'] = lambda: ModelCommitment(identifier='resnet50_less_variation_iteration=5', 
                                                                                activations_model=model.get_model('resnet50_less_variation_iteration=5'),
                                                                                  layers=MODEL_CONFIGS['resnet50_less_variation_iteration=5']['model_commitment']['layers'],
                                                                                    region_layer_map=MODEL_CONFIGS['resnet50_less_variation_iteration=5']["region_layer_map"])

model_registry['resnet50_no_variation_iteration=1'] = lambda: ModelCommitment(identifier='resnet50_no_variation_iteration=1',
                                                                               activations_model=model.get_model('resnet50_no_variation_iteration=1'),
                                                                                 layers=MODEL_CONFIGS['resnet50_no_variation_iteration=1']['model_commitment']['layers'],
                                                                                   region_layer_map=MODEL_CONFIGS['resnet50_no_variation_iteration=1']["region_layer_map"])

model_registry['resnet50_no_variation_iteration=2'] = lambda: ModelCommitment(identifier='resnet50_no_variation_iteration=2',
                                                                               activations_model=model.get_model('resnet50_no_variation_iteration=2'),
                                                                                 layers=MODEL_CONFIGS['resnet50_no_variation_iteration=2']['model_commitment']['layers'],
                                                                                   region_layer_map=MODEL_CONFIGS['resnet50_no_variation_iteration=2']["region_layer_map"])

model_registry['resnet50_no_variation_iteration=3'] = lambda: ModelCommitment(identifier='resnet50_no_variation_iteration=3', 
                                                                              activations_model=model.get_model('resnet50_no_variation_iteration=3'), 
                                                                                layers=MODEL_CONFIGS['resnet50_no_variation_iteration=3']['model_commitment']['layers'],
                                                                                  region_layer_map=MODEL_CONFIGS['resnet50_no_variation_iteration=3']["region_layer_map"])

model_registry['resnet50_no_variation_iteration=4'] = lambda: ModelCommitment(identifier='resnet50_no_variation_iteration=4', 
                                                                              activations_model=model.get_model('resnet50_no_variation_iteration=4'), 
                                                                              layers=MODEL_CONFIGS['resnet50_no_variation_iteration=4']['model_commitment']['layers'], 
                                                                              region_layer_map=MODEL_CONFIGS['resnet50_no_variation_iteration=4']["region_layer_map"])

model_registry['resnet50_no_variation_iteration=5'] = lambda: ModelCommitment(identifier='resnet50_no_variation_iteration=5', 
                                                                              activations_model=model.get_model('resnet50_no_variation_iteration=5'), 
                                                                              layers=MODEL_CONFIGS['resnet50_no_variation_iteration=5']['model_commitment']['layers'],
                                                                                region_layer_map=MODEL_CONFIGS['resnet50_no_variation_iteration=5']["region_layer_map"])

model_registry['resnet50_z_axis_iteration=1'] = lambda: ModelCommitment(identifier='resnet50_z_axis_iteration=1', 
                                                                        activations_model=model.get_model('resnet50_z_axis_iteration=1'), 
                                                                        layers=MODEL_CONFIGS['resnet50_z_axis_iteration=1']['model_commitment']['layers'], 
                                                                        region_layer_map=MODEL_CONFIGS['resnet50_z_axis_iteration=1']["region_layer_map"])

model_registry['resnet50_z_axis_iteration=2'] = lambda: ModelCommitment(identifier='resnet50_z_axis_iteration=2',
                                                                         activations_model=model.get_model('resnet50_z_axis_iteration=2'), 
                                                                         layers=MODEL_CONFIGS['resnet50_z_axis_iteration=2']['model_commitment']['layers'], 
                                                                         region_layer_map=MODEL_CONFIGS['resnet50_z_axis_iteration=2']["region_layer_map"])

model_registry['resnet50_z_axis_iteration=3'] = lambda: ModelCommitment(identifier='resnet50_z_axis_iteration=3',
                                                                         activations_model=model.get_model('resnet50_z_axis_iteration=3'), 
                                                                         layers=MODEL_CONFIGS['resnet50_z_axis_iteration=3']['model_commitment']['layers'], 
                                                                         region_layer_map=MODEL_CONFIGS['resnet50_z_axis_iteration=3']["region_layer_map"])

model_registry['resnet50_z_axis_iteration=4'] = lambda: ModelCommitment(identifier='resnet50_z_axis_iteration=4',
                                                                        activations_model=model.get_model('resnet50_z_axis_iteration=4'), 
                                                                        layers=MODEL_CONFIGS['resnet50_z_axis_iteration=4']['model_commitment']['layers'], 
                                                                        region_layer_map=MODEL_CONFIGS['resnet50_z_axis_iteration=4']["region_layer_map"])

model_registry['resnet50_z_axis_iteration=5'] = lambda: ModelCommitment(identifier='resnet50_z_axis_iteration=5', 
                                                                        activations_model=model.get_model('resnet50_z_axis_iteration=5'),
                                                                          layers=MODEL_CONFIGS['resnet50_z_axis_iteration=5']['model_commitment']['layers'],
                                                                            region_layer_map=MODEL_CONFIGS['resnet50_z_axis_iteration=5']["region_layer_map"])

model_registry['resnet50_wo_shading_iteration=1'] = lambda: ModelCommitment(identifier='resnet50_wo_shading_iteration=1',
                                                                             activations_model=model.get_model('resnet50_wo_shading_iteration=1'), 
                                                                             layers=MODEL_CONFIGS['resnet50_wo_shading_iteration=1']['model_commitment']['layers'], 
                                                                             region_layer_map=MODEL_CONFIGS['resnet50_wo_shading_iteration=1']["region_layer_map"])

model_registry['resnet50_wo_shading_iteration=2'] = lambda: ModelCommitment(identifier='resnet50_wo_shading_iteration=2', 
                                                                            activations_model=model.get_model('resnet50_wo_shading_iteration=2'), 
                                                                            layers=MODEL_CONFIGS['resnet50_wo_shading_iteration=2']['model_commitment']['layers'], 
                                                                            region_layer_map=MODEL_CONFIGS['resnet50_wo_shading_iteration=2']["region_layer_map"])

model_registry['resnet50_wo_shading_iteration=3'] = lambda: ModelCommitment(identifier='resnet50_wo_shading_iteration=3', 
                                                                            activations_model=model.get_model('resnet50_wo_shading_iteration=3'), 
                                                                            layers=MODEL_CONFIGS['resnet50_wo_shading_iteration=3']['model_commitment']['layers'], 
                                                                            region_layer_map=MODEL_CONFIGS['resnet50_wo_shading_iteration=3']["region_layer_map"])

model_registry['resnet50_wo_shading_iteration=4'] = lambda: ModelCommitment(identifier='resnet50_wo_shading_iteration=4', 
                                                                            activations_model=model.get_model('resnet50_wo_shading_iteration=4'), 
                                                                            layers=MODEL_CONFIGS['resnet50_wo_shading_iteration=4']['model_commitment']['layers'],
                                                                              region_layer_map=MODEL_CONFIGS['resnet50_wo_shading_iteration=4']["region_layer_map"])

model_registry['resnet50_wo_shading_iteration=5'] = lambda: ModelCommitment(identifier='resnet50_wo_shading_iteration=5', 
                                                                            activations_model=model.get_model('resnet50_wo_shading_iteration=5'), 
                                                                            layers=MODEL_CONFIGS['resnet50_wo_shading_iteration=5']['model_commitment']['layers'], 
                                                                            region_layer_map=MODEL_CONFIGS['resnet50_wo_shading_iteration=5']["region_layer_map"])

model_registry['resnet50_wo_shadows_iteration=1'] = lambda: ModelCommitment(identifier='resnet50_wo_shadows_iteration=1',
                                                                             activations_model=model.get_model('resnet50_wo_shadows_iteration=1'), 
                                                                             layers=MODEL_CONFIGS['resnet50_wo_shadows_iteration=1']['model_commitment']['layers'], 
                                                                             region_layer_map=MODEL_CONFIGS['resnet50_wo_shadows_iteration=1']["region_layer_map"])

model_registry['resnet50_wo_shadows_iteration=2'] = lambda: ModelCommitment(identifier='resnet50_wo_shadows_iteration=2',
                                                                             activations_model=model.get_model('resnet50_wo_shadows_iteration=2'),
                                                                               layers=MODEL_CONFIGS['resnet50_wo_shadows_iteration=2']['model_commitment']['layers'], 
                                                                               region_layer_map=MODEL_CONFIGS['resnet50_wo_shadows_iteration=2']["region_layer_map"])

model_registry['resnet50_wo_shadows_iteration=3'] = lambda: ModelCommitment(identifier='resnet50_wo_shadows_iteration=3', 
                                                                            activations_model=model.get_model('resnet50_wo_shadows_iteration=3'),
                                                                              layers=MODEL_CONFIGS['resnet50_wo_shadows_iteration=3']['model_commitment']['layers'], 
                                                                              region_layer_map=MODEL_CONFIGS['resnet50_wo_shadows_iteration=3']["region_layer_map"])

model_registry['resnet50_wo_shadows_iteration=4'] = lambda: ModelCommitment(identifier='resnet50_wo_shadows_iteration=4',
                                                                             activations_model=model.get_model('resnet50_wo_shadows_iteration=4'), 
                                                                             layers=MODEL_CONFIGS['resnet50_wo_shadows_iteration=4']['model_commitment']['layers'], 
                                                                             region_layer_map=MODEL_CONFIGS['resnet50_wo_shadows_iteration=4']["region_layer_map"])

model_registry['resnet50_wo_shadows_iteration=5'] = lambda: ModelCommitment(identifier='resnet50_wo_shadows_iteration=5', 
                                                                            activations_model=model.get_model('resnet50_wo_shadows_iteration=5'),
                                                                              layers=MODEL_CONFIGS['resnet50_wo_shadows_iteration=5']['model_commitment']['layers'], 
                                                                              region_layer_map=MODEL_CONFIGS['resnet50_wo_shadows_iteration=5']["region_layer_map"])

model_registry['resnet50_ambient_iteration=1'] = lambda: ModelCommitment(identifier='resnet50_ambient_iteration=1', 
                                                                         activations_model=model.get_model('resnet50_ambient_iteration=1'), 
                                                                         layers=MODEL_CONFIGS['resnet50_ambient_iteration=1']['model_commitment']['layers'], 
                                                                         region_layer_map=MODEL_CONFIGS['resnet50_ambient_iteration=1']["region_layer_map"])

model_registry['resnet50_ambient_iteration=2'] = lambda: ModelCommitment(identifier='resnet50_ambient_iteration=2', 
                                                                         activations_model=model.get_model('resnet50_ambient_iteration=2'), 
                                                                         layers=MODEL_CONFIGS['resnet50_ambient_iteration=2']['model_commitment']['layers'], 
                                                                         region_layer_map=MODEL_CONFIGS['resnet50_ambient_iteration=2']["region_layer_map"])

model_registry['resnet50_ambient_iteration=3'] = lambda: ModelCommitment(identifier='resnet50_ambient_iteration=3', 
                                                                         activations_model=model.get_model('resnet50_ambient_iteration=3'), 
                                                                         layers=MODEL_CONFIGS['resnet50_ambient_iteration=3']['model_commitment']['layers'], 
                                                                         region_layer_map=MODEL_CONFIGS['resnet50_ambient_iteration=3']["region_layer_map"])

model_registry['resnet50_ambient_iteration=4'] = lambda: ModelCommitment(identifier='resnet50_ambient_iteration=4', 
                                                                         activations_model=model.get_model('resnet50_ambient_iteration=4'), 
                                                                         layers=MODEL_CONFIGS['resnet50_ambient_iteration=4']['model_commitment']['layers'],
                                                                           region_layer_map=MODEL_CONFIGS['resnet50_ambient_iteration=4']["region_layer_map"])

model_registry['resnet50_ambient_iteration=5'] = lambda: ModelCommitment(identifier='resnet50_ambient_iteration=5',
                                                                          activations_model=model.get_model('resnet50_ambient_iteration=5'),
                                                                            layers=MODEL_CONFIGS['resnet50_ambient_iteration=5']['model_commitment']['layers'],
                                                                              region_layer_map=MODEL_CONFIGS['resnet50_ambient_iteration=5']["region_layer_map"])

model_registry['resnet50_textures_iteration=1'] = lambda: ModelCommitment(identifier='resnet50_textures_iteration=1', 
                                                                          activations_model=model.get_model('resnet50_textures_iteration=1'),
                                                                          layers=MODEL_CONFIGS['resnet50_textures_iteration=1']['model_commitment']['layers'],
                                                                            region_layer_map=MODEL_CONFIGS['resnet50_textures_iteration=1']["region_layer_map"])

model_registry['resnet50_textures_iteration=2'] = lambda: ModelCommitment(identifier='resnet50_textures_iteration=2',
                                                                           activations_model=model.get_model('resnet50_textures_iteration=2'),
                                                                             layers=MODEL_CONFIGS['resnet50_textures_iteration=2']['model_commitment']['layers'], 
                                                                             region_layer_map=MODEL_CONFIGS['resnet50_textures_iteration=2']["region_layer_map"])

model_registry['resnet50_textures_iteration=3'] = lambda: ModelCommitment(identifier='resnet50_textures_iteration=3', 
                                                                          activations_model=model.get_model('resnet50_textures_iteration=3'), 
                                                                          layers=MODEL_CONFIGS['resnet50_textures_iteration=3']['model_commitment']['layers'],
                                                                          region_layer_map=MODEL_CONFIGS['resnet50_textures_iteration=3']["region_layer_map"])

model_registry['resnet50_textures_iteration=4'] = lambda: ModelCommitment(identifier='resnet50_textures_iteration=4', 
                                                                          activations_model=model.get_model('resnet50_textures_iteration=4'),
                                                                          layers=MODEL_CONFIGS['resnet50_textures_iteration=4']['model_commitment']['layers'], 
                                                                          region_layer_map=MODEL_CONFIGS['resnet50_textures_iteration=4']["region_layer_map"])

model_registry['resnet50_textures_iteration=5'] = lambda: ModelCommitment(identifier='resnet50_textures_iteration=5', 
                                                                          activations_model=model.get_model('resnet50_textures_iteration=5'), 
                                                                          layers=MODEL_CONFIGS['resnet50_textures_iteration=5']['model_commitment']['layers'], 
                                                                          region_layer_map=MODEL_CONFIGS['resnet50_textures_iteration=5']["region_layer_map"])

model_registry['resnet50_no_specular_iteration=1'] = lambda: ModelCommitment(identifier='resnet50_no_specular_iteration=1',
                                                                              activations_model=model.get_model('resnet50_no_specular_iteration=1'), 
                                                                              layers=MODEL_CONFIGS['resnet50_no_specular_iteration=1']['model_commitment']['layers'], 
                                                                              region_layer_map=MODEL_CONFIGS['resnet50_no_specular_iteration=1']["region_layer_map"])

model_registry['resnet50_no_specular_iteration=2'] = lambda: ModelCommitment(identifier='resnet50_no_specular_iteration=2', 
                                                                             activations_model=model.get_model('resnet50_no_specular_iteration=2'), 
                                                                             layers=MODEL_CONFIGS['resnet50_no_specular_iteration=2']['model_commitment']['layers'],
                                                                               region_layer_map=MODEL_CONFIGS['resnet50_no_specular_iteration=2']["region_layer_map"])

model_registry['resnet50_no_specular_iteration=3'] = lambda: ModelCommitment(identifier='resnet50_no_specular_iteration=3', 
                                                                             activations_model=model.get_model('resnet50_no_specular_iteration=3'), 
                                                                             layers=MODEL_CONFIGS['resnet50_no_specular_iteration=3']['model_commitment']['layers'], 
                                                                             region_layer_map=MODEL_CONFIGS['resnet50_no_specular_iteration=3']["region_layer_map"])

model_registry['resnet50_no_specular_iteration=4'] = lambda: ModelCommitment(identifier='resnet50_no_specular_iteration=4', 
                                                                             activations_model=model.get_model('resnet50_no_specular_iteration=4'),
                                                                               layers=MODEL_CONFIGS['resnet50_no_specular_iteration=4']['model_commitment']['layers'], 
                                                                               region_layer_map=MODEL_CONFIGS['resnet50_no_specular_iteration=4']["region_layer_map"])

model_registry['resnet50_no_specular_iteration=5'] = lambda: ModelCommitment(identifier='resnet50_no_specular_iteration=5', 
                                                                             activations_model=model.get_model('resnet50_no_specular_iteration=5'),
                                                                            layers=MODEL_CONFIGS['resnet50_no_specular_iteration=5']['model_commitment']['layers'],
                                                                            region_layer_map=MODEL_CONFIGS['resnet50_no_specular_iteration=5']["region_layer_map"])
