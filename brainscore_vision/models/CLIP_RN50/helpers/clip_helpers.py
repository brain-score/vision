import clip
import functools
import torch
from .imagenet_class_names import imagenet_class_names
from brainscore_vision.model_helpers.activations.pytorch import load_images


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

  def forward(self, img):
    with torch.no_grad():
      image_features = self.clmodel.encode_image(img).float()
      image_features /= image_features.norm(dim=-1, keepdim=True)
      return self.logits(self.text_features, image_features)


