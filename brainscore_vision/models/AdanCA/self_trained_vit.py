import torch
import torch.nn as nn
from functools import partial
import torchvision

from timm.models.vision_transformer_nca import NCAFormer, _cfg
from timm.models.vision_transformer import VisionTransformer
from timm.models.swin_transformer import SwinTransformer, swin_small_patch4_window7_224, swin_base_patch4_window7_224, swin_tiny_patch4_window7_224
from timm.models.swin_transformer_v2 import swinv2_base_window8_256, swinv2_small_window8_256
from timm.models.robust_vit import rvt_base_plus, rvt_small_plus, rvt_base, rvt_small
from timm.models.tap_robust_vit import tap_rvt_base_plus
from timm.models.FAN.fan import fan_base_18_p16_224, fan_small_12_p16_224, fan_small_12_p4_hybrid, fan_base_16_p4_hybrid, fan_small_12_p4_hybrid
from timm.models.FAN.tap_fan import tap_fan_base_16_p4_hybrid
from timm.models.maxxvit import maxvit_small_tf_224
from timm.models.deit import deit_base_patch16_224
from timm.models.convit import convit_base
from timm.models.resnet import resnet50, resnet18
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models import create_model


__all__ = [
    'nca_vit_tiny_patch16_224', \
    'nca_vit_small_patch16_224', \
    'nca_vit_base_patch16_224'
]

@register_model
def nca_vit(args, pretrained=False, **kwargs) -> NCAFormer:
    params = {
        'patch_size': 4,
        'embed_dim': 96,
        'nca_norm':args.nca_norm,
        'drop_path_rate':args.dropout_path,
        'depth': args.depth,
        'num_heads': args.num_heads,
        'separate_norm':args.separate_norm,
        'stochastic_update':args.stochastic_update,
        'times':args.times,
        'alive_channel':args.alive_channel,
        'alive_threshold':args.alive_threshold,
        'trainable_kernel':args.trainable_kernel,
        'normalize_filter':args.normalize_filter,
        'padding_mode':args.padding_mode,
        'multi_head_perception':args.multi_head_perception,
        'perception_scales':args.perception_scales,
        'pos_emb':args.pos_emb,
        'perception_aggr':args.perception_aggr,
        'block_type':args.block_type,
        'learned_patch_merging':args.learned_patch_merging,
        'learn_patch_merging_alive_only':args.learn_patch_merging_alive_only,
        'residual_nca':args.residual_nca,
        'solver':args.solver,
        'head_with_alive':args.head_with_alive,
        'sigmoid_alive':args.sigmoid_alive,
        'energy_minimization':args.energy_minimization,
        'low_rank_approx':args.low_rank_approx,
        'multi_head_nca':args.multi_head_nca,
        'mlp_proj':args.mlp_proj,
        'ablation_nca':args.ablation_nca,
        'linear_downsample':args.linear_downsample,
        'cnn_front_end':args.cnn_front_end,
        'local_attn_v2':args.local_attn_v2,
        'local_attn_v2_block_type':args.local_attn_v2_block_type,
        'middle_linear_supervision':args.middle_linear_supervision,
        'linear_combine':args.linear_combine,
        'correct_alive':args.correct_alive,
        'no_global':args.no_global,
        'nca_str':args.nca_str,
        'recurrent_attention':args.recurrent_attention,
        'paas':args.paas,
        'weighted_combine':args.weighted_combine,
        'sparse_query':args.sparse_query,
        'sparse_query_method':args.sparse_query_method,
        'overlap_patch_embed':args.overlap_patch_embed,
        'group_norm':args.group_norm,
        'v2':args.v2,
        'recurrent_attention_norm':args.recurrent_attention_norm,
        'cosine_attn':args.cosine_attn,
        'energy_multi_head':args.energy_multi_head,
        'qk_norm':args.qk_norm,
        'energy_coeff_init':args.energy_coeff_init,
        'relative_pos_emb':args.relative_pos_emb,
        'window_attn_str':args.window_attn,
    }
    

    for key in params.keys():
        if key in kwargs:
            params[key] = kwargs.pop(key)
    print("NCA ViT config: ")
    for key, value in params.items():
        print(key, ":", value)

    model = NCAFormer(**params, **kwargs)
    
    # model.default_cfg = _cfg()
    if pretrained:
        model.load_state_dict(torch.load("pretrained/vit_tiny_patch16_224.pth"))
    # model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    # model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    params = {
        'patch_size': 16,
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
    }

    for key in params.keys():
        if key in kwargs:
            params[key] = kwargs.pop(key)

    model = VisionTransformer(**params, **kwargs)
    
    # model.default_cfg = _cfg()
    if pretrained:
        model.load_state_dict(torch.load("pretrained/vit_tiny_patch16_224.pth"))
    # model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    # model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_small_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Small (ViT-S/16)
    """
    params = {
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
    }

    for key in params.keys():
        if key in kwargs:
            params[key] = kwargs.pop(key)

    model = VisionTransformer(**params, **kwargs)
    # model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    if pretrained:
        model.load_state_dict(torch.load("pretrained/vit_small_patch16_224.pth"))
    # model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    params = {
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
    }

    for key in params.keys():
        if key in kwargs:
            params[key] = kwargs.pop(key)

    model = VisionTransformer(**params, **kwargs)
    # model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    if pretrained:
        model.load_state_dict(torch.load("pretrained/vit_base_patch16_224.pth"))
    return model

@register_model
def swin_tiny_patch16_224(pretrained=False, **kwargs) -> SwinTransformer:
    # patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)
    img_size = kwargs.get('img_size',224)
    if(img_size == 224):
        
        params = {
            'patch_size': 4,
            'window_size': 7,
            'embed_dim': 96,
            'depths': (2, 2, 6, 2),
            'num_heads': (3, 6, 12, 24),
        }
    elif(img_size == 32):
        params = {
            'patch_size': 2,
            'window_size': 4,
            'embed_dim': 96,
            'depths': (2, 6, 2),
            'num_heads': (3, 6, 12),
        }

    for key in params.keys():
        if key in kwargs:
            params[key] = kwargs.pop(key)

    model = SwinTransformer(**params, **kwargs)
    return model
@register_model
def swin_small_patch16_224(pretrained=False, **kwargs) -> SwinTransformer:
    # patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)
    img_size = kwargs.get('img_size',224)
    if(img_size == 224):
        params = {
            'patch_size': 4,
            'window_size': 7,
            'embed_dim': 96,
            'depths': (2, 2, 18, 2),
            'num_heads': (3, 6, 12, 24),
        }
    elif(img_size == 32):
        params = {
            'patch_size': 2,
            'window_size': 4,
            'embed_dim': 96,
            'depths': (2, 2, 18, 2),
            'num_heads': (3, 6, 12, 24),
        }

    for key in params.keys():
        if key in kwargs:
            params[key] = kwargs.pop(key)
    model = SwinTransformer(**params, **kwargs)
    return model
@register_model
def swin_base_patch16_224(pretrained=False, **kwargs) -> SwinTransformer:
    # patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)
    params = {
        'patch_size': 4,
        'window_size': 7,
        'embed_dim': 128,
        'depths': (2, 2, 18, 2),
        'num_heads': (4, 8, 16, 32),
    }

    for key in params.keys():
        if key in kwargs:
            params[key] = kwargs.pop(key)

    model = SwinTransformer(**params, **kwargs)
    return model

def get_models(args):
    print("Build Models")
    img_size = 224
    n_class = 1000
    if(args.data == "cifar10"):
        img_size = 32
        n_class = 10
    if(args.data == "cifar100"):
        img_size = 32
        n_class = 100
    if(args.data == "tiny-imagenet"):
        img_size = 64
        n_class = 200
    if(args.data == "imagenet100"):
        img_size = 224
        n_class = 100
    model_args = dict(patch_size=args.patch_size, img_size = img_size, num_classes = n_class)
    if(args.avg_pool and "swin" not in args.model):
        model_args.update(dict(global_pool = "avg", class_token = False))
    if(args.embed_dim != 0):
        model_args.update(dict(embed_dim = args.embed_dim))
        
    if(args.model == "tiny"):
        model = vit_tiny_patch16_224(pretrained = args.from_pretrained, **model_args)
    elif(args.model == "small"):
        model = vit_small_patch16_224(pretrained = args.from_pretrained, **model_args)
    elif(args.model == "base"):
        model = vit_base_patch16_224(pretrained = args.from_pretrained, **model_args)
    elif(args.model == 'nca_vit'):
        model = nca_vit(args, pretrained = args.from_pretrained, **model_args)
    elif(args.model == "swin_tiny"):
        model_args.update(dict(drop_path_rate = 0.1))
        params = {
                'noisy_attn':args.noisy_attn,
                'binary_attn':args.binary_attn,
            }
        model_args.update(params)
        model = swin_tiny_patch16_224(pretrained = args.from_pretrained, **model_args)
    elif(args.model == "swin_tiny_ori"):
        model = swin_tiny_patch16_224(pretrained = args.from_pretrained, drop_path_rate = 0.1, num_classes = n_class)
    elif(args.model == "swin_small"):
        model_args.update(dict(drop_path_rate = 0.3))
        model = swin_small_patch16_224(pretrained = args.from_pretrained, **model_args)
    elif(args.model == "swin_base"):
        #if(args.data == "imagenet"):
        model_args.update(dict(drop_path_rate = 0.5))
        model = swin_base_patch16_224(pretrained = args.from_pretrained, **model_args)
    elif(args.model == "swin_base_abl"):
        # FLOPs + 1.25G
        model_args.update(dict(drop_path_rate = 0.5))
        model = swin_base_patch4_window7_224(pretrained = False, num_classes = n_class, depths=(2, 2, 20, 2))
    elif(args.model == "rvt_small_plus"):
        model = rvt_small_plus(pretrained = args.from_pretrained, drop_path_rate = 0.1, num_classes = n_class)
    elif(args.model == "fan_small_hybrid"):
        if(args.data == "imagenet100"):
            model = fan_small_12_p4_hybrid(drop_path_rate = 0.25, num_classes = n_class)
        else:
            model = fan_small_12_p4_hybrid(drop_path_rate = 0.25, num_classes = n_class)
    elif(args.model == "swinv2_small"):
        model_args.update(dict(drop_path_rate = 0.3))
        model = swinv2_small_window8_256(pretrained = args.from_pretrained, **model_args)
    elif(args.model == "swinv2_base"):
        model_args.update(dict(drop_path_rate = 0.5, img_size = 256))
        model = swinv2_base_window8_256(pretrained = args.from_pretrained, **model_args)
    elif(args.model == "resnet50_torch"):
        model = torchvision.models.resnet50(num_classes = n_class)
    elif(args.model == "resnet50"):
        model = resnet50(pretrained = args.from_pretrained, num_classes = n_class)
    elif(args.model == "resnet18"):
        model = resnet18(pretrained = args.from_pretrained, num_classes = n_class)
    elif(args.model == "deit_base"):
        model = deit_base_patch16_224(pretrained = args.from_pretrained, drop_path_rate = 0.1, num_classes = n_class)
    elif(args.model == "deit_base_pool"):
        model = deit_base_patch16_224(pretrained = args.from_pretrained, drop_path_rate = 0.1, global_pool = "avg", class_token = False, num_classes = n_class)
    elif(args.model == "deit_base_abl"):
        model = deit_base_patch16_224(pretrained = args.from_pretrained, drop_path_rate = 0.1, depth = 13, num_classes = n_class)
    elif(args.model == "convit_base_abl"):
        model = convit_base(pretrained = args.from_pretrained, drop_path_rate = 0.1, depth = 13, num_classes = n_class)
    elif("nca_version" in args.model):
        params = {
                'nca_model':args.nca_model,
                'before_nca_norm':args.before_nca_norm,
                'stochastic_update':args.stochastic_update,
                'times':args.times,
                'energy_minimization':args.energy_minimization,
                'weighted_scale_combine':args.weighted_scale_combine,
                'nca_local_perception':args.nca_local_perception,
                'nca_local_perception_loop':args.nca_local_perception_loop,
                'energy_point_wise':args.energy_point_wise,
                'nca_expand':args.nca_expand,
                'init_with_grad_kernel':args.init_with_grad_kernel,
                'perception_norm':args.nca_perception_norm,
            }
        model_args.update(params)
        if("swin_tiny" in args.model):
            if(args.ablation_unroll):
                model_args.update(dict(ablation_unroll = True))
            if(args.ablation_wsum):
                model_args.update(dict(ablation_wsum = True))
            if(args.ablation_msp):
                model_args.update(dict(ablation_msp = args.ablation_msp))
            if(args.ablation_aggrnorm != "bn"):
                model_args.update(dict(ablation_aggrnorm = args.ablation_aggrnorm))
            model_args.update(dict(drop_path_rate = 0.1))
            model = swin_tiny_patch16_224(pretrained = args.from_pretrained, **model_args)
        if("swin_base" in args.model):
            model_args.update(dict(drop_path_rate = 0.5))
            model = swin_base_patch16_224(pretrained = args.from_pretrained, **model_args)
        if("rvt_small_plus" in args.model):
            model = rvt_small_plus(pretrained = args.from_pretrained, drop_path_rate = 0.1, num_classes = n_class, **params)
        if("rvt_base_plus" in args.model):
            model = rvt_base_plus(pretrained = args.from_pretrained, drop_path_rate = 0.1, num_classes = n_class, **params)
        if("fan_small_hybrid" in args.model):
            params.update(dict(adl_loss = args.adl_loss_weight > 0.0))
            if(args.ablation_unroll):
                params.update(dict(ablation_unroll = True))
            if(args.ablation_wsum):
                params.update(dict(ablation_wsum = True))
            if(args.ablation_msp):
                params.update(dict(ablation_msp = args.ablation_msp))
            if(args.data == "imagenet100"):
                model = fan_small_12_p4_hybrid(drop_path_rate = 0.25, num_classes = n_class, **params)
            else:
                model = fan_small_12_p4_hybrid(drop_path_rate = 0.25, num_classes = n_class, **params)
        if("fan_base_hybrid" in args.model):
            params.update(dict(adl_loss = args.adl_loss_weight > 0.0))
            if("pool" in args.model):
                model = fan_base_16_p4_hybrid(drop_path_rate = 0.35, num_classes = n_class, avg_pool = 1, **params)
            else:
                model = fan_base_16_p4_hybrid(drop_path_rate = 0.35, num_classes = n_class, **params)
        if("swinv2_base" in args.model):
            model_args.update(dict(drop_path_rate = 0.5, img_size = 256))
            model = swinv2_base_window8_256(pretrained = args.from_pretrained, **model_args)
        if("maxvit_small" in args.model):
            model = maxvit_small_tf_224(pretrained = args.from_pretrained, drop_path_rate = 0.3, num_classes = n_class, **params)
        if("fan_base_vit" in args.model):
            model = fan_base_18_p16_224(drop_path_rate = 0.35, num_classes = n_class, **params)
        if("deit_base" in args.model):
            model = deit_base_patch16_224(pretrained = args.from_pretrained, drop_path_rate = 0.1, num_classes = n_class, **params) #, global_pool = "avg", class_token = False
        if("convit_base" in args.model):
            model = convit_base(pretrained = args.from_pretrained, drop_path_rate = 0.1, num_classes = n_class, **params)
    else:
        if("maxvit_small" in args.model and "." not in args.model):
            model = maxvit_small_tf_224(pretrained = args.from_pretrained, drop_path_rate = 0.3, num_classes = n_class)
        elif(args.model == "rvt_base_plus"):
            model = rvt_base_plus(pretrained = args.from_pretrained, drop_path_rate = 0.1, num_classes = n_class)
        elif(args.model == "rvt_small_plus"):
            model = rvt_small_plus(pretrained = args.from_pretrained, num_classes = n_class)
        elif(args.model == "rvt_small"):
            model = rvt_small(pretrained = args.from_pretrained, num_classes = n_class)
        elif(args.model == "rvt_base"):
            model = rvt_base(pretrained = args.from_pretrained, num_classes = n_class)
        elif(args.model == "tap_rvt_base_plus" ):
            model = tap_rvt_base_plus(pretrained = args.from_pretrained)
        elif(args.model == "tap_fan_base"):
            model = tap_fan_base_16_p4_hybrid(drop_path_rate = 0.35, num_classes = n_class, return_attn = args.adl_loss_weight > 0.0)
            if(args.from_pretrained):
                print("Loading Pre-Trained TAP FAN")
                model.load_state_dict(torch.load("pretrained_models/tapadl_fan_base.pth.tar")["state_dict"])
        else:
            if("swin" in args.model and "." not in args.model and "v2" not in args.model):
                print("Swin from TIMM lib")
                if("swin_base" in args.model):
                    model = swin_base_patch4_window7_224(pretrained = args.from_pretrained, num_classes = n_class)
                elif("swin_small" in args.model):
                    model = swin_small_patch4_window7_224(pretrained = args.from_pretrained, num_classes = n_class)
            elif("swinv2" in args.model and "." not in args.model):
                if("base" in args.model):
                    if(args.data == "imagenet"):
                        model = swinv2_base_window8_256(pretrained = args.from_pretrained, num_classes = n_class, drop_path_rate = 0.5)
                    else:
                        model = swinv2_base_window8_256(pretrained = args.from_pretrained, num_classes = n_class, drop_path_rate = 0.5)
            elif("fan" in args.model):
                print("FAN from InterNet")
                if(args.model == "fan_base_vit"):
                    model = fan_base_18_p16_224()
                    if(args.from_pretrained):
                        print("Loading Pre-Trained FAN ViT")
                        model.load_state_dict(torch.load("pretrained_models/fan_vit_base.pth.tar"))
                elif(args.model == "fan_small_vit"):
                    model = fan_small_12_p16_224()
                    model.load_state_dict(torch.load("pretrained_models/fan_vit_small.pth.tar"))
                elif(args.model == "fan_small_hybrid"):
                    model = fan_small_12_p4_hybrid()
                    model.load_state_dict(torch.load("pretrained_models/fan_hybrid_small.pth.tar"))
                elif(args.model == "fan_base_hybrid"):
                    model = fan_base_16_p4_hybrid(drop_path_rate = 0.35, num_classes = n_class, adl_loss = args.adl_loss_weight > 0.0)
                    if(args.from_pretrained):
                        print("Loading Pre-Trained FAN")
                        model.load_state_dict(torch.load("pretrained_models/fan_hybrid_base.pth.tar"))
                elif(args.model == "fan_base_hybrid_pool"):
                    model = fan_base_16_p4_hybrid(drop_path_rate = 0.35, num_classes = n_class, avg_pool = 1)
            else:
                print("Model from timm repo")
                model = create_model(args.model, pretrained = args.from_pretrained)
                print(model.default_cfg)
    return model