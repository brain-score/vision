import functools

import torch
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_preprocess_images
from PIL import Image
import numpy as np
import timm
import logging
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from model_tools.activations.core import ActivationsExtractorHelper
from model_tools.brain_transformation import ModelCommitment
from model_tools.utils import fullname
from model_tools.brain_transformation import ModelCommitment, LayerScores, LayerMappedModel, LayerSelection, \
    RegionLayerMap, STANDARD_REGION_BENCHMARKS
import torch.nn as nn
from brainio.stimuli import StimulusSet
from albumentations import (
    Compose, Normalize, Resize,CenterCrop
    )
from albumentations.pytorch import ToTensorV2
# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models

import os 

image_resize = 324
image_crop = 288
norm_mean = [0.485, 0.456, 0.406] 
norm_std = [0.229, 0.224, 0.225]
freeze_layers = ['blocks.0.0', 'blocks.0.1', 'blocks.1.0', 
                'blocks.1.1', 'blocks.1.2', 'blocks.2.0', 
                'blocks.2.1', 'blocks.2.2', 'blocks.3.0', 'blocks.3.1', 'blocks.3.2']
layers = [
    #'blocks', 'blocks.0', 'blocks.0.0', 'blocks.0.1', 
    #'blocks.1', 'blocks.1.0', 'blocks.1.1', 'blocks.1.2', 
    #'blocks.2', 'blocks.2.0', 'blocks.2.1', 'blocks.2.2', 
    #'blocks.3', 'blocks.3.0', 'blocks.3.1', 'blocks.3.2', 'blocks.3.3',
    'blocks.4', 'blocks.4.0', 
    #'blocks.4.0.conv_pw', 'blocks.4.0.conv_dw', 'blocks.4.0.conv_pwl', 'blocks.4.1', 'blocks.4.1.conv_pw', 'blocks.4.1.conv_dw', 'blocks.4.1.conv_pwl', 'blocks.4.2', 
    #'blocks.4.2.conv_pw', 'blocks.4.2.conv_dw', 'blocks.4.2.conv_pwl', 'blocks.4.3', 'blocks.4.3.conv_pw', 'blocks.4.3.conv_dw', 'blocks.4.3.conv_pwl', 'blocks.5', 
    #'blocks.5.0', 'blocks.5.0.conv_pw', 'blocks.5.0.conv_dw', 'blocks.5.0.conv_pwl', 'blocks.5.1', 'blocks.5.1.conv_pw', 'blocks.5.1.conv_dw', 'blocks.5.1.conv_pwl', 
    #'blocks.5.2', 'blocks.5.2.conv_pw', 'blocks.5.2.conv_dw', 'blocks.5.2.conv_pwl', 'blocks.5.3', 'blocks.5.3.conv_pw', 'blocks.5.3.conv_dw', 'blocks.5.3.conv_pwl', 
    #'blocks.5.4', 'blocks.5.4.conv_pw', 'blocks.5.4.conv_dw', 'blocks.5.4.conv_pwl', 'blocks.6', 'blocks.6.0', 'blocks.6.0.conv_pw', 'blocks.6.0.conv_dw', 
    #'blocks.6.0.conv_pwl', 'blocks.6.1', 'blocks.6.1.conv_pw', 'blocks.6.1.conv_dw', 'blocks.6.1.conv_pwl',
    #'global_pool', 'global_pool.flatten', 'global_pool.pool'
    ]
def custom_image_preprocess(images, **kwargs):
    
    transforms_val = Compose([
        Resize(image_resize, image_resize),
        CenterCrop(image_crop, image_crop),
        Normalize(mean=norm_mean,std=norm_std,),
        ToTensorV2()])
    
    images = [np.array(pillow_image) for pillow_image in images]
    images = [transforms_val(image=image)["image"] for image in images]
    images = np.stack(images)

    return images

def load_preprocess_images_custom(image_filepaths, preprocess_images=custom_image_preprocess,  **kwargs):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = preprocess_images(images, **kwargs)
    return images

def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper()\
                and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image

def get_model_list():
    return ['effnetb1_cutmixpatch_augmix_robust32_avge4e7_ONELAYER_TTA_324x288']


class EffNetBX(nn.Module):
    def __init__(self,):
        super().__init__ ()
        self.efnet_model = timm.create_model('tf_efficientnet_b1_ns', pretrained=True)

    def forward(self, x):
        x = self.efnet_model(x)
        return x



class TTAModelCommitment(ModelCommitment):
    """
    Similar to ModelCommitment but gets model activations multiple times depending on the number of trials. To be
    used with models that have stochastic activations.
    """

    def __init__(self, identifier, activations_model, layers, visual_degrees=8):
            
            layer_selection = StochasticLayerSelection(model_identifier=identifier,
                                                       activations_model=activations_model, layers=layers,
                                                       visual_degrees=visual_degrees)
            region_layer_map = RegionLayerMap(layer_selection=layer_selection,
                                              region_benchmarks=STANDARD_REGION_BENCHMARKS)
            
            super(TTAModelCommitment, self).__init__(identifier=identifier, activations_model=activations_model,
                                                            layers=layers, region_layer_map=region_layer_map,
                                                            visual_degrees=visual_degrees)      
    def look_at(self, stimuli, number_of_trials=4):
        stimuli_identifier = stimuli.identifier
        for trial_number in range(number_of_trials):
            print(f"[TTACOMMIT] TRIAL {stimuli.identifier} NUMBER {trial_number}")
            if stimuli_identifier:
                stimuli.identifier = stimuli_identifier + '-trial' + f'{trial_number:03d}'
            if trial_number == 0:
                activations = super().look_at(stimuli, number_of_trials=1)
                if not activations.values.flags['WRITEABLE']:
                    activations.values.setflags(write=1)
            else:
                activations += super().look_at(stimuli, number_of_trials=1)
        stimuli.identifier = stimuli_identifier
        return activations/number_of_trials

class StochasticLayerSelection(LayerSelection):
    def __init__(self, model_identifier, activations_model, layers,  visual_degrees):
        super(StochasticLayerSelection, self).__init__(model_identifier=model_identifier,
                                                       activations_model=activations_model, layers=layers,
                                                       visual_degrees=visual_degrees)
        self._layer_scoring = StochasticLayerScores(model_identifier=model_identifier,
                                                    activations_model=activations_model,
                                                    visual_degrees=visual_degrees)


class StochasticLayerScores(LayerScores):
    def _create_mapped_model(self, region, layer, model, model_identifier, visual_degrees):
        return StochasticLayerMappedModel(identifier=f"{model_identifier}-{layer}", visual_degrees=visual_degrees,
                                activations_model=model, region_layer_map={region: layer})


class StochasticLayerMappedModel(LayerMappedModel):
    def run_activations(self, stimuli, layers, number_of_trials=1):
        stimuli_identifier = stimuli.identifier
        for trial_number in range(number_of_trials):
            print(f"[StochasticLayer] TRIAL {stimuli.identifier} NUMBER {trial_number}")
            if stimuli_identifier:
                stimuli.identifier = stimuli_identifier + '-trial' + f'{trial_number:03d}'
            if trial_number == 0:
                activations = self.activations_model(stimuli, layers=layers)
            else:
                activations += self.activations_model(stimuli, layers=layers)
        stimuli.identifier = stimuli_identifier
        return activations / number_of_trials


SUBMODULE_SEPARATOR = '.'
from collections import OrderedDict

class TTAPytorchWrapper(PytorchWrapper):
    def __init__(self, model, preprocessing, identifier=None, *args, **kwargs):
        super(TTAPytorchWrapper, self).__init__(model, preprocessing, identifier=None, *args, **kwargs)

        
    def get_activations(self, images, layer_names):
        import torch
        import random
        from torch.autograd import Variable
        images = [torch.from_numpy(image) for image in images]
        images = Variable(torch.stack(images))
        tta = random.choice([0,1,2,3])
        if tta == 0:
            images = images
        if tta == 1:
            images = images.flip(-2)
        if tta == 2:
            images = images.flip(-1)
        if tta == 3:
            images = images.flip(-2).flip(-1)
        #print(tta)
        #print(images.shape)
        images = images.to(self._device)
        self._model.eval()

        layer_results = OrderedDict()
        hooks = []
        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            hook = self.register_hook(layer, layer_name, target_dict=layer_results)
            hooks.append(hook)

        self._model(images)
        for hook in hooks:
            hook.remove()
        return layer_results

    def get_layer(self, layer_name):
        if layer_name == 'logits':
            return self._output_layer()
        module = self._model
        for part in layer_name.split(SUBMODULE_SEPARATOR):
            module = module._modules.get(part)
            assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
        return module





def get_model(name):
    assert name == 'effnetb1_cutmixpatch_augmix_robust32_avge4e7_ONELAYER_TTA_324x288'
    model_tf_efficientnet_b1_ns= EffNetBX()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_tf_efficientnet_b1_ns.load_state_dict(torch.load(dir_path + "/tf_efficientnet_b1_ns_robust_cutmixpatchresize_augmix_e4toe7.pth", map_location=torch.device('cpu'))["model"])
    model = model_tf_efficientnet_b1_ns.efnet_model
    filter_elems = set(["se", "act", "bn", "conv"])
    layer_list = [layer for layer, _ in model.named_modules() if not any(i in layer for i in filter_elems)]
    print(layer_list)
    print(len(layer_list))
    
    for n, m in model.named_modules():
      if isinstance(m, nn.BatchNorm2d) and any(x in n for x in ["conv_stem" ] + freeze_layers) or n =="bn1":
        print(f"Freeze {n, m}")
        m.eval()
    
    
    preprocessing = functools.partial(load_preprocess_images_custom, 
                                        preprocess_images=custom_image_preprocess,
                                        )


    activations_model  = TTAPytorchWrapper(identifier='effnetb1_cutmixpatch_augmix_robust32_avge4e7_ONELAYER_TTA_324x288', model=model, preprocessing=preprocessing, batch_size=8)
    model = TTAModelCommitment(identifier='effnetb1_cutmixpatch_augmix_robust32_avge4e7_ONELAYER_TTA_324x288', activations_model=activations_model ,
                        # specify layers to consider
                        layers=layers)
    #model.layer_model.region_layer_map['V1'] = 'module.vone_block.output'

    return model


def get_bibtex(model_identifier):
    return """
    @article {Dapello2020.06.16.154542,
	author = {Dapello, Joel and Marques, Tiago and Schrimpf, Martin and Geiger, Franziska and Cox, David D. and DiCarlo, James J.},
	title = {Simulating a Primary Visual Cortex at the Front of CNNs Improves Robustness to Image Perturbations},
	elocation-id = {2020.06.16.154542},
	year = {2020},
	doi = {10.1101/2020.06.16.154542},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/10/22/2020.06.16.154542},
	eprint = {https://www.biorxiv.org/content/early/2020/10/22/2020.06.16.154542.full.pdf},
	journal = {bioRxiv}
                """


if __name__ == '__main__':
    check_models.check_brain_models(__name__)
