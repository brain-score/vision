from torchvision import transforms
from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper


def get_transform_videos(transform_img):
    def transform_videos(videos):
        import torch
        ret = []
        for video in videos:
            frames = []
            for frame in video.to_pil_imgs():
                frames.append(transform_img(frame))
            ret.append(torch.stack(frames))
        ret = torch.stack(ret)
        return ret.permute(0, 2, 1, 3, 4)
    return transform_videos


def get_resnet_base_models(identifier):
    img_transform = transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    vid_transform = get_transform_videos(img_transform)

    if identifier == "R3D":
        spec = {
            "input": {
                "type": "video",
                "fps": 25
            },
            "activation": {
                "stem": "CTHW",
                **{f'layer{i}': "CTHW" for i in range(1, 5)},
                "avgpool": "C",
                "fc": "C"
            },
            "model": {
                "objective": "ACTION_RECOSGNITION",
                "dataset": "KINETICS_400",
                "source": "torchvision",
                "architecture": ("RESNET", "Conv3D"),
                "acc@1": 63.2,
                "acc@5": 83.479,
                "params": 33.4e6,
                "gflops": 40.7
            },
        }
        model_name = "r3d_18"

    from torchvision.models import video as vid
    model = getattr(vid, model_name)(weights="KINETICS400_V1")

    return PytorchWrapper(identifier, spec, model, vid_transform)


if __name__ == "__main__":
    model = get_resnet_base_models("R3D")
    print(model)
    print(model.spec)

    