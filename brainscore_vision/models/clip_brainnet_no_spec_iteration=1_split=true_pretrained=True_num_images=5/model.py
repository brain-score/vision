from brainscore_vision.model_helpers.check_submission import check_models
import functools
import numpy as np
import torch
from transformers import AutoFeatureExtractor, CvtForImageClassification, CLIPVisionModel, CLIPProcessor, CLIPModel
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from PIL import Image

from open_clip.transformer import VisionTransformer
import torch
import open_clip
from torch import nn
device = "cpu"
import pytorch_lightning as pl

from brainscore_vision.model_helpers.check_submission import check_models
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
import torch
import numpy as np
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
import torchvision.models as models
import gdown

# This is an example implementation for submitting custom model named my_custom_model

# Attention: It is important, that the wrapper identifier is unique per model!
# The results will otherwise be the same due to brain-scores internal result caching mechanism.
# Please load your pytorch model for usage in CPU. There won't be GPUs available for scoring your model.
# If the model requires a GPU, contact the brain-score team directly.
class CosineLRScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_length, steps, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_length = warmup_length
        self.steps = steps
        super(CosineLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_length:
            lr = self.base_lr * (self.last_epoch + 1) / self.warmup_length
        else:
            e = self.last_epoch - self.warmup_length
            es = self.steps - self.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.base_lr
        return [lr]
    
class CLIPLightning(pl.LightningModule):
    def __init__(self, ver="ViT-B-16", data="datacomp_l_s1b_b8k", **kwargs) -> None:
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(ver, pretrained=data)
        self.model = model.to(device)  # Move model to GPU
        self.vision_model: VisionTransformer = model.visual
        self.vision_model.requires_grad_(False)
        self.vision_model.eval()
        self.tokenizer = open_clip.get_tokenizer(ver)
        self.num_classs = num_classes

    def forward(self, images, texts):
        images = images.to(device)
        texts = texts.to(device)
        image_features = self.model.encode_image(images.to(device))  # Move images to GPU
        texts = [f'{dict_class_names[int(texts[i])]}' for i in range(len(texts))]
        texts = self.tokenizer(texts)
        text_features = self.model.encode_text(texts.to(device))  # Move texts to GPU
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.mean()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)
        top_labels = torch.tensor(top_labels)
        return logits_per_image, logits_per_text

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Forward pass
        logits_per_image, logits_per_text = self.forward(images, labels)
        # Compute loss
        target_labels = torch.tensor(labels).to(device)  # Move labels to GPU
        labels = torch.arange(len(images), dtype=torch.long, device=device)
        loss = self.compute_loss(logits_per_image, logits_per_text, labels)
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # Define validation_step, test_step, compute_loss, and configure_optimizers as before
    def validation_step(self, batch, batch_idx):
            images, labels = batch
            # Forward pass
            logits_per_image, logits_per_text = self.forward(images, labels)
            # Compute loss
            target_labels = torch.tensor(labels)
            labels = torch.arange(len(images),dtype=torch.long,device=device)
            loss = self.compute_loss(logits_per_image, logits_per_text, labels)
            # Log loss
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
    def test_step(self, batch, batch_idx):
            images, labels = batch
            # Forward pass
            logits_per_image, logits_per_text = self.forward(images, labels)
            # Compute loss
            target_labels = torch.tensor(labels)
            labels = torch.arange(len(images),dtype=torch.long,device=device)
            loss = self.compute_loss(logits_per_image, logits_per_text, labels)
            # Log loss
            self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
        
    def compute_loss(self, logits_per_image,  logits_per_text, ground_truth):
            loss_img = nn.CrossEntropyLoss()
            loss_txt = nn.CrossEntropyLoss()
            # Compute contrastive loss
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            # Total loss
            return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2*1e-2, weight_decay=3e-3, betas=(0.9, 0.95), eps=6.5e-09)
        scheduler = {
            'scheduler': CosineLRScheduler(optimizer, 1e-3,warmup_length=1824, steps=2000),
            'monitor': 'val_loss'  # Adjust the monitored quantity
        }
        return [optimizer], [scheduler]

def get_bibtex(model_identifier):
    return """xx"""


def get_model_list():
  return ['clip_brainnet_no_spec_iteration=1_split=true_pretrained=True_num_images=5']

def get_model(name):
    assert name == 'clip_brainnet_no_spec_iteration=1_split=true_pretrained=True_num_images=5'
    # https://huggingface.co/models?sort=downloads&search=cvt
    image_size = 224
    url = "https://drive.google.com/file/d/1WE5yGmEini1rpOiVNAjEFlCbVT5qge-u/view?usp=drive_link"
    output = "clip_brainnet_no_spec_iteration=1_split=true_pretrained=True_num_images=5.ckpt"
    gdown.download(url, output)

    # Wrap the model in PytorchWrapper directly
    model = CLIPLightning.load_from_checkpoint(output)
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)

    return activations_model


def get_layers(name):
    assert name == 'clip_brainnet_no_spec_iteration=1_split=true_pretrained=True_num_images=5'
    layers = []
    url = "https://drive.google.com/file/d/1WE5yGmEini1rpOiVNAjEFlCbVT5qge-u/view?usp=drive_link"
    output = "clip_brainnet_no_spec_iteration=1_split=true_pretrained=True_num_images=5.ckpt"
    gdown.download(url, output)
    model = CLIPLightning.load_from_checkpoint(name)
    for name, module in model.named_modules():
        if 'vision_model.encoder.layers' in name:
            layers.append(name)
    # Añadir las capas del modelo q CLIPVisionModel
    #layers += ['vision_model.encoder.layers'+str(i) for i in range(1)]
    # Ejemplo: puedes elegir algunas capas específicas si lo deseas
    # layers = ['vision_model.encoder.layers.3', 'vision_model.post_layernorm']

    return layers

if __name__ == '__main__':
    # Use this method to ensure the correctness of the BaseModel implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_base_models(__name__)
