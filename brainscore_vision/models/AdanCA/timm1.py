import sys
# sys.path.append("/home/ytang/workspace/image_models/")
# sys.path.append("/home/ytang/workspace/brain-score/brainscore_vision/model_helpers/activations/")

import os
# os.environ["RESULTCACHING_HOME"] = r"/home/ytang/workspace/data/cache/.resultcaching"
# os.environ["BRAINIO_HOME"] = r"/home/ytang/workspace/data/cache/.brainio"
# os.environ["BRAINSCORE_HOME"] = r"/home/ytang/workspace/data/cache/.brain-score"
# os.environ['TORCH_HOME'] = r'/home/ytang/workspace/data/cache/.torch'

from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore.utils import LazyLoad
from collections import OrderedDict
from torchvision import transforms


import timm
# import tome
import torch
import gdown
import numpy as np
from timm.models.robust_vit import rvt_base_plus, rvt_small_plus
from timm.models.tap_robust_vit import tap_rvt_base_plus
from timm.models.FAN.fan import fan_base_18_p16_224, fan_small_12_p16_224, fan_small_12_p4_hybrid, fan_base_16_p4_hybrid, fan_small_12_p4_hybrid
from timm.models.FAN.tap_fan import tap_fan_base_16_p4_hybrid
from torchvision.transforms.functional import InterpolationMode


def get_model(name):

    if("resnet" in name):
        from resnet import ResNet18
        
        model = ResNet18()
        pretrained_file = torch.load(f"pretrained_models/resnet.pth", map_location="cpu")
        model_weight = pretrained_file["net"]
        new_state_dict = {}
        for k, v in model_weight.items():
            if k.startswith('module.'):
                # Remove the 'module.' prefix
                new_name = k[7:]  # remove `module.`
            else:
                new_name = k
            new_state_dict[new_name] = v
        model.load_state_dict(new_state_dict)
        model.eval()

        transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        def transform(paths):
            from PIL import Image
            import torch
            images = [Image.open(path) for path in paths]
            images = [transform_test(image) for image in images]
            images = torch.stack(images, dim=0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            return images
        return model, transform

    if("timm_model" in name):
        

        if("vit_base_patch16_224" in name):
            
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
            
            if("TOME" in name):
                
                # Set the number of tokens reduced per layer. See paper for details.
                if("linear" in name):
                    tome.patch.timm(model)
                    model.r = [32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0]
                    if("_1" in name):
                        model.r = [1,1,1,1,1,1, 0, 0, 0, 0, 0, 0]
                elif("random" in name):
                    tome.patch.timm(model, random = True)
                    model.r = 16
                else:
                    tome.patch.timm(model)
                    model.r = int(name.split("_")[-1])

        else:
            crop_pct = 1.0
            input_size = 224
            if("swin_base" in name):
                crop_pct = 0.875
                model = timm.create_model("swin_base_patch4_window7_224.ms_in1k", pretrained=True)
            if("swin_small" in name):
                crop_pct = 0.875
                model = timm.create_model("swin_small_patch4_window7_224.ms_in1k", pretrained=True)
            if("swin_tiny" in name):
                crop_pct = 0.875
                model = timm.create_model("swin_tiny_patch4_window7_224.ms_in1k", pretrained=True)
            if("swinv2_base" in name):
                crop_pct = 0.875
                input_size = 256
                model = timm.create_model("swinv2_base_window8_256.ms_in1k", pretrained=True)
            if("fan_" in name):
                model = fan_base_16_p4_hybrid(drop_path_rate = 0.35)
                # https://drive.google.com/file/d/1Y8s_HXSySZeAjLk9Q7TBe_OwZUaNXhR6/view?usp=sharing
                url = 'https://drive.google.com/uc?id=1Y8s_HXSySZeAjLk9Q7TBe_OwZUaNXhR6'
                output = f'./pretrained_models/fan_hybrid_base.pth.tar'
                gdown.download(url, output, quiet=False)
                model.load_state_dict(torch.load("pretrained_models/fan_hybrid_base.pth.tar", map_location="cpu"))
            if("rvt_" in name):
                model = rvt_base_plus(pretrained = False)
                # https://drive.google.com/file/d/1G4UOqgnw2YvEI0dSaiYkxG1JE6-Bd-qe/view?usp=sharing
                url = 'https://drive.google.com/uc?id=1G4UOqgnw2YvEI0dSaiYkxG1JE6-Bd-qe'
                output = f'./pretrained_models/rvt_base_plus_self_trained_imagenet.pth.tar'
                gdown.download(url, output, quiet=False)
                pretrained_file = torch.load(f"pretrained_models/rvt_base_plus_self_trained_imagenet.pth.tar", map_location="cpu")
                model_weight = pretrained_file["model"]
                model.load_state_dict(model_weight)
            if("convit" in name):
                model = timm.create_model("convit_base", pretrained=True)
        model.eval()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform_vit = transforms.Compose([
            transforms.Resize(int(input_size / crop_pct), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
        def transform(paths):
            from PIL import Image
            import torch
            images = [Image.open(path) for path in paths]
            images = [transform_vit(image) for image in images]
            images = torch.stack(images, dim=0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            return images
    elif("self_trained" in name):
        """Load args and models"""
        from timm.models.vision_transformer_nca import NCAFormer
        from timm.models.vision_transformer import VisionTransformer
        from timm.models.swin_transformer import SwinTransformer
        from self_trained_vit import get_models
        if("swin_" in name):
            url = 'https://drive.google.com/uc?id=12nLoEvqMMZac3g2qNec_JXncHz1xlnQH'
        elif("rvt_" in name):
            # https://drive.google.com/file/d/1M3FbvrgzuONol4vvYf7IVhqzTtnrlpEp/view?usp=sharing
            url = 'https://drive.google.com/uc?id=1M3FbvrgzuONol4vvYf7IVhqzTtnrlpEp'
        elif("fan_" in name):
            # https://drive.google.com/file/d/1pAXF-rBkDMH8-t_NTf9IuXN2eaATdxZa/view?usp=sharing
            url = 'https://drive.google.com/uc?id=1pAXF-rBkDMH8-t_NTf9IuXN2eaATdxZa'
        elif("convit" in name):
            # https://drive.google.com/file/d/13HQoXLmX_TwauFXAwA1KdxQ6qno-syUd/view?usp=sharing
            url = 'https://drive.google.com/uc?id=13HQoXLmX_TwauFXAwA1KdxQ6qno-syUd'
        output = f'./pretrained_models/{name}.pth.tar'
        gdown.download(url, output, quiet=False)
        pretrained_file = torch.load(f"pretrained_models/{name}.pth.tar", map_location="cpu")
        model_weight = pretrained_file["model"]
        args = pretrained_file["args"]
        if(hasattr(args, "adl_loss_weight")):
            args.adl_loss_weight = 0.0
        model = get_models(args)
        model.load_state_dict(model_weight)
        model.eval()

        crop_pct = 1.0
        input_size = 224
        if("swin" in name):
            crop_pct = 0.875
        if("swinv2_base" in name):
            crop_pct = 0.875
            input_size = 256


        # if("imagenet" in args.data):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform_test = transforms.Compose([
            transforms.Resize(int(input_size / crop_pct), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])
        # else:
        #     transform_test = transforms.Compose([
        #         transforms.Resize(32, interpolation=InterpolationMode.BICUBIC),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        #     ])

        def transform(paths):
            from PIL import Image
            import torch
            images = [Image.open(path) for path in paths]
            images = [transform_test(image) for image in images]
            images = torch.stack(images, dim=0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            return images
    # elif("rvt_" in name):
    #     if(name == "rvt_base_plus"):
    #         model = rvt_base_plus(pretrained = True)
    #     elif(name == "rvt_small_plus"):
    #         model = rvt_small_plus(pretrained = True)
    #     model.eval()


    #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])
    #     transform_test = transforms.Compose([
    #         transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])


    #     def transform(paths):
    #         from PIL import Image
    #         import torch
    #         images = [Image.open(path) for path in paths]
    #         images = [transform_test(image) for image in images]
    #         images = torch.stack(images, dim=0)
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         images = images.to(device)
    #         return images
        

    return model, transform

def get_middle_layer(model, input_x, layers, model_name):
    import torch
    model.eval()
    # input_x = [torch.from_numpy(image) if not isinstance(image, torch.Tensor) else image for image in input_x]
    # input_x = torch.stack(input_x)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_x = input_x.to(device)
    with torch.no_grad():
        if(len(layers) > 0):
            if("nca_" in layers[0]):
                # import pickle
                # with open("input.pkl", "wb") as f:
                #     pickle.dump(input_x, f, pickle.HIGHEST_PROTOCOL)
                output, return_dict = model(input_x, return_extra = True)
            else:
                output, return_dict = model(input_x, return_middle_state = True)
    # print(return_dict.keys(), layers)
    multi_layer_dict = OrderedDict()
    for layer in layers:
        """The shape examination is for brainscore test. Seems the library cannot handle activations with first transformer-like output and later conv-like output"""
        if(len(return_dict[layer].shape) == 4):
            b, c, h, w = return_dict[layer].shape
            if(h == w):
                data = return_dict[layer].permute(0, 2, 3, 1).reshape(b, h*w, c)
            elif(c == h):
                b,h,w,c = return_dict[layer].shape
                data = return_dict[layer].reshape(b, h*w, c)
        else:
            data = return_dict[layer]
        multi_layer_dict[layer] = data.detach()
    return multi_layer_dict

class TIMMWrapper_dict(PytorchWrapper):
    def __init__(self, model, transform, model_name):
        print("#################")
        print(model_name)
        self.model_name = model_name
        print("#################")
        super().__init__(model, transform, identifier=model_name)
    
    def get_activations(self, images, layer_names):
        return get_middle_layer(self._model, images, layer_names, self.model_name)

class TIMMWrapper(PytorchWrapper):
    def __init__(self, model, transform, model_name):
        print("#################")
        print(model_name)
        print("#################")
        super().__init__(model, transform, identifier=model_name)
    
    def get_activations(self, images, layer_names):
        normal_output = super().get_activations(images, layer_names)
        normal_output_correct = OrderedDict()
        for key in normal_output.keys():
            data = normal_output[key]
            # Check if the data has four dimensions
            if data.ndim == 4:
                b, c, h, w = data.shape
                if h == w:
                    # Permute dimensions and reshape
                    data = np.transpose(data, (0, 2, 3, 1)).reshape(b, h*w, c)
                elif c == h:
                    # Reshape directly without permutation
                    data = data.reshape(b, h*w, c)
            # Assign the manipulated data back to the dictionary
            normal_output_correct[key] = data
        return normal_output_correct


class TIMMWrapper_combine(PytorchWrapper):
    def __init__(self, model, transform, model_name):
        print("#################")
        print(model_name)
        self.model_name = model_name
        print("#################")
        super().__init__(model, transform, identifier=model_name)
    
    def get_activations(self, images, layer_names):
        layer_names_nca = [x for x in layer_names if "nca_" in x]
        layer_names_not_nca = [x for x in layer_names if "nca_" not in x]
        images = [torch.from_numpy(image) if not isinstance(image, torch.Tensor) else image for image in images]
        images = torch.stack(images)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)
        normal_output = super().get_activations(images, layer_names_not_nca)

        normal_output_correct = OrderedDict()
        for key in normal_output.keys():
            data = normal_output[key]
            # Check if the data has four dimensions
            if data.ndim == 4:
                b, c, h, w = data.shape
                if h == w:
                    # Permute dimensions and reshape
                    data = np.transpose(data, (0, 2, 3, 1)).reshape(b, h*w, c)
                elif c == h:
                    # Reshape directly without permutation
                    data = data.reshape(b, h*w, c)
            # Assign the manipulated data back to the dictionary
            normal_output_correct[key] = data
        if(len(layer_names_nca) > 0):
            nca_output = get_middle_layer(self._model, images, layer_names_nca, self.model_name)
            normal_output_correct.update(nca_output)
        return normal_output_correct


pool = {}
# for r in range(1, 17):
#     pool[f"VIT_B_16_224_TOME{r}"] = LazyLoad(lambda: TIMMWrapper(*get_model(f"vit_base_patch16_224_TOME_{r}"), f"VIT_B_16_224_TOME{r}"))

pool = {
    "VIT_B_16_224": LazyLoad(lambda: TIMMWrapper(*get_model("vit_base_patch16_224"), "VIT_B_16_224")),
    "VIT_B_16_224_TOME16": LazyLoad(lambda: TIMMWrapper(*get_model("vit_base_patch16_224_TOME_timm_model_16"), "VIT_B_16_224_TOME16")),
    "VIT_B_16_224_TOME12": LazyLoad(lambda: TIMMWrapper(*get_model("vit_base_patch16_224_TOME_timm_model_12"), "VIT_B_16_224_TOME12")),
    "VIT_B_16_224_TOME8": LazyLoad(lambda: TIMMWrapper(*get_model("vit_base_patch16_224_TOME_timm_model_8"), "VIT_B_16_224_TOME8")),
    "VIT_B_16_224_TOME4": LazyLoad(lambda: TIMMWrapper(*get_model("vit_base_patch16_224_TOME_timm_model_4"), "VIT_B_16_224_TOME4")),
    "VIT_B_16_224_TOME1": LazyLoad(lambda: TIMMWrapper(*get_model("vit_base_patch16_224_TOME_timm_model_1"), "VIT_B_16_224_TOME1")),
    "VIT_B_16_224_TOME0": LazyLoad(lambda: TIMMWrapper(*get_model("vit_base_patch16_224_TOME_timm_model_0"), "VIT_B_16_224_TOME0")),
    "VIT_B_16_224_TOME_random": LazyLoad(lambda: TIMMWrapper(*get_model("vit_base_patch16_224_TOME_random_timm_model"), "vit_base_patch16_224_TOME_random")),
    "VIT_B_16_224_TOME_linear": LazyLoad(lambda: TIMMWrapper(*get_model("vit_base_patch16_224_TOME_linear_timm_model"), "vit_base_patch16_224_TOME_linear")),
    "VIT_B_16_224_TOME_linear_1": LazyLoad(lambda: TIMMWrapper(*get_model("vit_base_patch16_224_TOME_linear_1_timm_model"), "vit_base_patch16_224_TOME_linear_1")),
    "deit_small_patch16_224": LazyLoad(lambda: TIMMWrapper(*get_model("deit_small_patch16_224"), "deit_small_patch16_224")),
    "swin_small_patch4_window7_224": LazyLoad(lambda: TIMMWrapper(*get_model("swin_small_patch4_window7_224_timm_model"), "swin_small_patch4_window7_224")),
    "swinv2_base": LazyLoad(lambda: TIMMWrapper(*get_model("swinv2_base_timm_model"), "swinv2_base")),
    
    "swin_base_patch4_window12_384": LazyLoad(lambda: TIMMWrapper(*get_model("swin_base_patch4_window12_384_timm_model"), "swin_base_patch4_window12_384")),
    "swin_tiny_patch4_window7_224": LazyLoad(lambda: TIMMWrapper(*get_model("swin_tiny_patch4_window7_224_timm_model"), "swin_tiny_patch4_window7_224")),
    "swin_tiny_self_trained_cifar10": LazyLoad(lambda: TIMMWrapper(*get_model("swin_tiny_self_trained_cifar10"), "swin_tiny_self_trained_cifar10")),
    "swin_small_self_trained_cifar10": LazyLoad(lambda: TIMMWrapper(*get_model("swin_small_self_trained_cifar10"), "swin_small_self_trained_cifar10")),
    "vit_tiny_self_trained_cifar10": LazyLoad(lambda: TIMMWrapper(*get_model("tiny_self_trained_cifar10"), "vit_tiny_self_trained_cifar10")),
    "vit_small_self_trained_cifar10": LazyLoad(lambda: TIMMWrapper(*get_model("small_self_trained_cifar10"), "vit_small_self_trained_cifar10")),
    "swin_tiny_self_trained_imagenet100": LazyLoad(lambda: TIMMWrapper(*get_model("swin_tiny_self_trained_imagenet100"), "swin_tiny_self_trained_imagenet100")),
    "swin_small_self_trained_imagenet100": LazyLoad(lambda: TIMMWrapper(*get_model("swin_small_self_trained_imagenet100"), "swin_small_self_trained_imagenet100")),
    "swin_base_self_trained_imagenet100": LazyLoad(lambda: TIMMWrapper(*get_model("swin_base_self_trained_imagenet100"), "swin_base_self_trained_imagenet100")),
    "swin_base_self_trained_imagenet": LazyLoad(lambda: TIMMWrapper(*get_model("swin_base_self_trained_imagenet"), "swin_base_self_trained_imagenet")),

    "swin_base_nca_version_self_trained_imagenet": LazyLoad(lambda: TIMMWrapper_combine(*get_model("swin_base_nca_version_self_trained_imagenet"), "swin_base_nca_version_self_trained_imagenet")),
    "convit_base_nca_version_self_trained_imagenet": LazyLoad(lambda: TIMMWrapper_combine(*get_model("convit_base_nca_version_self_trained_imagenet"), "convit_base_nca_version_self_trained_imagenet")),
    "rvt_base_plus_nca_version_self_trained_imagenet": LazyLoad(lambda: TIMMWrapper_combine(*get_model("rvt_base_plus_nca_version_self_trained_imagenet"), "rvt_base_plus_nca_version_self_trained_imagenet")),
    "fan_base_hybrid_nca_version_self_trained_imagenet": LazyLoad(lambda: TIMMWrapper_combine(*get_model("fan_base_hybrid_nca_version_self_trained_imagenet"), "fan_base_hybrid_nca_version_self_trained_imagenet")),
    "swin_base_patch4_window7_224": LazyLoad(lambda: TIMMWrapper(*get_model("swin_base_patch4_window7_224_timm_model"), "swin_base_patch4_window7_224")),
    "convit_base": LazyLoad(lambda: TIMMWrapper(*get_model("convit_base_timm_model"), "convit_base")),
    "rvt_base_plus": LazyLoad(lambda: TIMMWrapper(*get_model("rvt_base_plus_timm_model"), "rvt_base_plus")),
    "fan_base_hybrid": LazyLoad(lambda: TIMMWrapper(*get_model("fan_base_hybrid_timm_model"), "fan_base_hybrid")),

    "vit_tiny_self_trained_imagenet100": LazyLoad(lambda: TIMMWrapper(*get_model("tiny_self_trained_imagenet100"), "vit_tiny_self_trained_imagenet100")),
    "vit_small_self_trained_imagenet100": LazyLoad(lambda: TIMMWrapper(*get_model("small_self_trained_imagenet100"), "vit_small_self_trained_imagenet100")),
    "nca_vit_self_trained_wtalivemask_cifar10": LazyLoad(lambda: TIMMWrapper_dict(*get_model("nca_vit_self_trained_wtalivemask_cifar10"), "nca_vit_self_trained_wtalivemask_cifar10")),
    "nca_vit_self_trained_noalivemask_cifar10": LazyLoad(lambda: TIMMWrapper_dict(*get_model("nca_vit_self_trained_noalivemask_cifar10"), "nca_vit_self_trained_noalivemask_cifar10")),
    "nca_vit_self_trained_wtalivemask_imagenet100": LazyLoad(lambda: TIMMWrapper_dict(*get_model("nca_vit_self_trained_wtalivemask_imagenet100"), "nca_vit_self_trained_wtalivemask_imagenet100")),
    "nca_vit_self_trained_wtalivemask_imagenet100_vone": LazyLoad(lambda: TIMMWrapper_dict(*get_model("nca_vit_self_trained_wtalivemask_imagenet100_vone"), "nca_vit_self_trained_wtalivemask_imagenet100_vone")),
    "nca_vit_self_trained_noalivemask_imagenet100": LazyLoad(lambda: TIMMWrapper_dict(*get_model("nca_vit_self_trained_noalivemask_imagenet100"), "nca_vit_self_trained_noalivemask_imagenet100")),
    "resnet18_cifar10": LazyLoad(lambda: TIMMWrapper_dict(*get_model("resnet18_cifar10"), "resnet18_cifar10")),
}
# pool.update(pool1)

if(__name__ == "__main__"):
    import torch
    model, _ = get_model("vit_base_patch16_224_TOME_16")
    dummy = torch.randn(5, 3, 224, 224)
    out = model(dummy)
    print(out.shape)