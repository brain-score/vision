import clip
import functools
import torch
from .imagenet_class_names import imagenet_class_names

from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.pytorch import load_images


# This is an example implementation for submitting alexnet as a pytorch model
# If you use pytorch, don't forget to add it to the setup.py

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
from model_tools.check_submission import check_models


def _load_and_preprocess(img, process_function):
    images = load_images(img)
    images = [process_function(image).numpy() for image in images]
    return images


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CosineSimilarityLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.matmul(a, b.T).T


class TransposingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(1, 0, 2)


class ClipModel(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()

        clmodel, preprocess = clip.load(architecture)
        self.clmodel = clmodel.eval().to(DEVICE)
        self.preprocessing = functools.partial(_load_and_preprocess, process_function=preprocess)

        text_descriptions = ["A photo of a " + label for label in imagenet_class_names]
        text_tokens = clip.tokenize(text_descriptions).to(DEVICE)
        with torch.no_grad():
            self.text_features = self.clmodel.encode_text(text_tokens).float()
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        self.logits = CosineSimilarityLayer()

        if architecture == "ViT-B/32":
            for resblock in self.clmodel.visual.transformer.resblocks:
                resblock.mlp.add_module("reshaper", TransposingLayer())
                resblock.mlp.add_module("invert_reshaper", TransposingLayer())

    def forward(self, img):
        with torch.no_grad():
            image_features = self.clmodel.encode_image(img).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return self.logits(self.text_features, image_features)


def get_model_list():
    return ["CLIP-ViT-B/32"]


def get_model(name):
    clip_model = ClipModel(name.strip("CLIP-"))
    wrapper = PytorchWrapper(
        identifier=name,
        model=clip_model,
        preprocessing=clip_model.preprocessing
    )
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    if name == "CLIP-RN50":
        return [
            'clmodel.visual.layer2.0.relu',
            'clmodel.visual.layer2.1.relu',
            'clmodel.visual.layer2.2.relu',
            'clmodel.visual.layer2.3.relu',
            'clmodel.visual.layer3.0.relu',
            'clmodel.visual.layer3.1.relu',
            'clmodel.visual.layer3.2.relu',
            'clmodel.visual.layer3.3.relu',
            'clmodel.visual.layer3.4.relu',
            'clmodel.visual.layer4.0.relu',
        ]
    elif name == "CLIP-ViT-B/32":
        return [f"clmodel.visual.transformer.resblocks.{i}.mlp.reshaper" for i in range(0, 12)]
    else:
        raise ValueError("Model not found.")


def get_bibtex(model_identifier):
    return """
@article{radford2021learning,
title={Learning transferable visual models from natural language supervision},
author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
journal={arXiv preprint arXiv:2103.00020},
year={2021}
}"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)
