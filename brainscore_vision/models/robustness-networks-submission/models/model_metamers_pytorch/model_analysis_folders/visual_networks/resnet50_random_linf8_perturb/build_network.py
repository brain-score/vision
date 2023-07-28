import sys
from robustness import datasets
from robustness.attacker import AttackerModel
from robustness.model_utils import make_and_restore_model
from model_analysis_folders.all_model_info import IMAGENET_PATH, MODEL_BASE_PATH
import os 

def build_net(ds_kwargs={}, return_metamer_layers=False):
    # We need to build the dataset so that the number of classes and normalization 
    # is set appropriately. You do not need to use this data for eval/metamer generation

    # Resnet50 Layers Used for Metamer Generation
    metamer_layers = [
         'input_after_preproc',
#          'conv1_relu1',
         'conv1_relu1_fake_relu',
#          'layer1',
         'layer1_fake_relu',
#          'layer2',
         'layer2_fake_relu',
#          'layer3',
         'layer3_fake_relu',
#          'layer4',
         'layer4_fake_relu',
         'avgpool',
         'final'
    ]

    ds = datasets.ImageNet(IMAGENET_PATH)

    ckpt_path = os.path.join(MODEL_BASE_PATH, 'visual_networks', 'pytorch_checkpoints', 'resnet50_random_linf8_perturb.pt')
    model, _ = make_and_restore_model(arch='resnet50', dataset=ds, resume_path=ckpt_path,
                                      pytorch_pretrained=False, parallel=False)
    

    # send the model to the GPU and return it. 
#     model.cuda()
    model.eval()
    if return_metamer_layers:
        return model, ds, metamer_layers
    else:
        return model, ds

def main(return_metamer_layers=False,
         ds_kwargs={}):
    if return_metamer_layers: 
        model, ds, metamer_layers = build_net(
                                              return_metamer_layers=return_metamer_layers,
                                              ds_kwargs=ds_kwargs)
        return model, ds, metamer_layers

    else:
        model, ds = build_net(
                              return_metamer_layers=return_metamer_layers,
                              ds_kwargs=ds_kwargs)
        return model, ds


if __name__== "__main__":
    main()
