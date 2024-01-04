import torch

from torch import nn

def get_clip_vision_model(clip_loss_net):
    state_dict_loaded = torch.jit.load(f'checkpoints/{clip_loss_net}.pt', map_location="cpu").eval().state_dict()  # RN50

    counts: list = [len(set(k.split(".")[2] for k in state_dict_loaded if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
    vision_layers = tuple(counts)
    vision_width = state_dict_loaded["visual.layer1.0.conv1.weight"].shape[0]
    output_width = round((state_dict_loaded["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    vision_patch_size = None
    assert output_width ** 2 + 1 == state_dict_loaded["visual.attnpool.positional_embedding"].shape[0]
    image_resolution = int(output_width * 32)
    embed_dim = state_dict_loaded["text_projection"].shape[1]
    vision_heads = vision_width * 32 // 64

    state_dict_loaded = {k[len('visual.'):]: v for k, v in state_dict_loaded.items() if k.startswith('visual.')}
    # state_dict_loaded = {k[len('visual.'):]: v.float() for k, v in state_dict_loaded.items() if k.startswith('visual.')}
    
    from .model import ModifiedResNet
    # bbn = ModifiedResNet(layers=(3, 4, 6, 3), output_dim=1024 , heads=32, input_resolution=224, width=64)
    bbn = ModifiedResNet(layers=vision_layers, output_dim=embed_dim , heads=vision_heads, input_resolution=image_resolution, width=vision_width)
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

    # bbn.apply(_convert_weights_to_fp16)

    bbn.load_state_dict(state_dict_loaded)
    
    # img_xfm_norm = NormalizeBatch((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # CLIP
    # bbn.float()
    # bbn.apply(_convert_weights_to_fp32)
    return bbn


"""
def get_my_model():
    from .model import MyModel
    return MyModel()
"""