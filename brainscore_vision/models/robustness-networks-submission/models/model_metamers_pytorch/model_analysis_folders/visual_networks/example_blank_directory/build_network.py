import sys
from robustness import datasets
from robustness.attacker import AttackerModel
from robustness.model_utils import make_and_restore_model
from model_analysis_folders.all_model_info import IMAGENET_PATH

def build_net(ds_kwargs={}, return_metamer_layers=False):
    # List of keys in the `all_outputs` dictionary. 
    metamer_layers = [
         'input_after_preproc', # Used for visualization purposes
         ## TODO: Add additional layers here, matching the layers used in the architecture dictionary
         'final' # classifier layer if it exists. Otherwise, a placeholder. 
    ]

    # We need to build the dataset so that the number of classes and normalization
    # is set appropriately. You do not need to use this data for metamer generation
    # TODO (optional): If there are additional dataset loading parameters for loading the model (ie it doesn't 
    # use the default cropping size) then include them here as `aug_train` and `aug_test` parameters for evaluation
    ds = datasets.ImageNet(IMAGENET_PATH)

    # TODO: If not included in the architecture file, specify the model checkpoint to use for loading. 
    ckpt_path = None

    # TODO: Specify the model name
    # Load the model architeccture from the robustness repo. Look in robustness.imagenet_models 
    # for examples. These should have the ability to return the `all_outputs` dictionary.
    model, _ = make_and_restore_model(arch='<MODEL_ARCHITECTURE_NAME>', dataset=ds, resume_path=ckpt_path,
                                      pytorch_pretrained=True, parallel=False, strict=True)


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
