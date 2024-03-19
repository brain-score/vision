from torchvision import transforms
from torchvision.models import video as vid

from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper


def get_transform_video(transform_img):
    def transform_video(video):
        import torch
        frames = torch.Tensor(video.to_numpy() / 255.0).permute(0, 3, 1, 2)
        frames = transform_img(frames)
        return frames.permute(1, 0, 2, 3)
    return transform_video


def get_model(identifier):
    if identifier in ["r3d_18", "r2plus1d_18", "mc3_18"]:
        img_transform = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
        inferencer_kwargs = {
            "fps": 25,
            "layer_activation_format": 
            {
                "stem": "CTHW",
                **{f'layer{i}': "CTHW" for i in range(1, 5)},
                "avgpool": "CTHW",
                "fc": "C"
            },
        }
        process_output = None

    elif identifier == "s3d":
        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
        inferencer_kwargs = {
            "fps": 15,
            "layer_activation_format":
            {
                **{f"features.{i}": "CTHW" for i in range(16)},
                "avgpool": "CTHW",
                "classifier": "CTHW"
            }
        }
        process_output = None

    elif identifier in ["mvit_v1_b", "mvit_v2_s"]:
        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
        inferencer_kwargs = {
            "fps": 7.5,
            "num_frames": 16,
            "layer_activation_format": {
                "conv_proj": "CTHW",
                **{f"blocks.{i}": "THWC" for i in range(16)},
                "head": "C",
            }
        }

        def process_output(layer, layer_name, input, output):
            if layer_name.startswith("blocks"):
                output, thw = output
                t, h, w = thw
                output = output[:, 1:]  # remove cls 
                b, n, c = output.shape
                assert n == t*h*w
                output = output.view(b, t, h, w, c)
                return output
            return output

    vid_transform = get_transform_video(img_transform)
    model_name = identifier
    model = getattr(vid, model_name)(weights="KINETICS400_V1")
    wrapper = PytorchWrapper(identifier, model, vid_transform, 
                             process_output=process_output,
                             **inferencer_kwargs)
    
    return wrapper