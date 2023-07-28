import sys
from robustness import datasets
from robustness.attacker import AttackerModel
from robustness.model_utils import make_and_restore_model
import torch
torch.backends.cudnn.benchmark = True

# Make a custom build script for audio_rep_training_cochleagram_1/l2_p1_robust_training
def build_net(ds_kwargs={}, return_metamer_layers=False):
    # We need to build the dataset so that the number of classes and normalization 
    # is set appropriately. You do not need to use this data for eval/metamer generation

    # Resnet50 Layers Used for Metamer Generation
    metamer_layers = [
#          'input_after_preproc',
         'patch_embed',
         'pos_embed',
         'block_0',
         'block_6',
         'block_12',
         'block_18',
         'blocks_end',
         'final'
    ]

    ds = datasets.ImageNet('/om2/data/public/imagenet/images_complete/ilsvrc/')
    # TODO: Get other augmentations? 

#     ckpt_path = '../pytorch_checkpoints/resnet50_byol.pt'
    ckpt_path = None
    model, _ = make_and_restore_model(arch='vit_large_patch16_224', dataset=ds, resume_path=ckpt_path,
                                      pytorch_pretrained=True, parallel=False, strict=True, 
                                      )

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
