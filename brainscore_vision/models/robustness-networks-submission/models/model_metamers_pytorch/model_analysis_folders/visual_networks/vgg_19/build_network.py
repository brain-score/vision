import sys
from robustness import datasets
from robustness.attacker import AttackerModel
from robustness.model_utils import make_and_restore_model
from model_analysis_folders.all_model_info import IMAGENET_PATH, MODEL_BASE_PATH

def build_net(ds_kwargs={}, return_metamer_layers=False):
    # We need to build the dataset so that the number of classes and normalization 
    # is set appropriately. You do not need to use this data for eval/metamer generation

    # VGG Layers Used for Metamer Generation
    metamer_layers = [
         'input_after_preproc',
#          'conv_relu_0_1',
         'conv_relu_0_1_fake_relu',
#          'maxpool_0',
#          'conv_relu_1_1',
         'conv_relu_1_1_fake_relu',
#          'maxpool_1',
#          'conv_relu_2_3',
         'conv_relu_2_3_fake_relu',
#          'maxpool_2',
#          'conv_relu_3_3',
         'conv_relu_3_3_fake_relu',
#          'maxpool_3',
#          'conv_relu_4_3',
         'conv_relu_4_3_fake_relu',
#          'maxpool_4',
         'avgpool',
#          'fc_relu_0',
         'fc_relu_0_fake_relu',
#          'fc_relu_1',
         'fc_relu_1_fake_relu',
         'final'
    ]

    ds = datasets.ImageNet(IMAGENET_PATH)

    model, _ = make_and_restore_model(arch='vgg19', dataset=ds, 
                                      pytorch_pretrained=True, parallel=False)

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
