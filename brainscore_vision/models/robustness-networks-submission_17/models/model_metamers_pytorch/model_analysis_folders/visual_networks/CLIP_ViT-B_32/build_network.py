import os
import sys
from robustness import datasets
from robustness.attacker import AttackerModel
from robustness.model_utils import make_and_restore_model
# sys.path.append('/om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/robustness/imagenet_models/clip')
from robustness.imagenet_models.clip import clip
# import clip
import torch 
from PIL import Image
SCRIPT_DIR = os.path.abspath(__file__)
sys.path.append(os.path.dirname(SCRIPT_DIR))
from imagenet_info_clip import imagenet_templates, imagenet_classes
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from tqdm import tqdm

def imagenet2_labelmap(classes, class_to_idx):
    class_to_idx = {'%s'%i:i for i in range(1000)}
    classes = ['%s'%i for i in range(1000)]
    return classes, class_to_idx

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

class CLIPModelWithLabels(torch.nn.Module):
    def __init__(self, clip_model):
        super(CLIPModelWithLabels, self).__init__()
        self.clip_model = clip_model
        self.input_resolution = clip_model.visual.input_resolution
        self.vision_embedding = clip_model.visual
        if os.path.exists('zeroshot_weights.pt'):
            zeroshot_weights = torch.load('zeroshot_weights.pt')
        else:
            zeroshot_weights = self.zeroshot_classifier(self.clip_model,
                                                             imagenet_classes,
                                                             imagenet_templates)
            torch.save(zeroshot_weights, 'zeroshot_weights.pt')
        self.register_buffer('zeroshot_weights', zeroshot_weights)

    def zeroshot_classifier(self, model, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates] #format with class
                if torch.cuda.is_available():
                    texts = clip.tokenize(texts).cuda() #tokenize
                else:
                    texts = clip.tokenize(texts)# .cuda() #tokenize
                class_embeddings = model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        return zeroshot_weights

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        if with_latent:
            image_features, _, all_outputs = self.vision_embedding(x, 
                                                 with_latent=with_latent,
                                                 fake_relu=fake_relu,
                                                 no_relu=no_relu)
            all_outputs['final_image_features'] = image_features
        else:
            image_features = self.vision_embedding(x)
        image_features /= image_features.norm(dim=-1, keepdim=True).detach()
        logits = 100. * image_features @ self.zeroshot_weights
        if with_latent:
            all_outputs['final'] = logits
            all_outputs['logits'] = logits
            return logits, None, all_outputs
        return logits
    
def _convert_image_to_rgb(image):
    return image.convert("RGB")

# Make a custom build script for audio_rep_training_cochleagram_1/l2_p1_robust_training
def build_net(ds_kwargs={}, return_metamer_layers=False):
    # We need to build the dataset so that the number of classes and normalization 
    # is set appropriately. You do not need to use this data for eval/metamer generation

    # Resnet50 Layers Used for Metamer Generation
    metamer_layers = [
         'input_after_preproc',
         'conv1',
         'class_embedding_cat',
         'ln_pre', 
         'block_0',
         'block_3', 
         'block_6',
         'block_9',
         'blocks_end',
         'final'
    ]


#     ckpt_path = '../pytorch_checkpoints/clip_rn50.pt'
#     model, _ = make_and_restore_model(arch='resnet50', dataset=ds, resume_path=ckpt_path,
#                                       pytorch_pretrained=False, parallel=False, strict=True, 
#                                       )

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
#     model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), jit=False)
    model.to(device)

#     image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
#     print(image)
#     text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
#     
#         logits_per_image, logits_per_text = model(image, text)
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    model = CLIPModelWithLabels(model)

    n_px = model.input_resolution
    transforms = Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor()])

    # Check with ImageNetV2 because that is the posted accuracy for the checkpoint. 
    # images = torchvision.datasets.ImageFolder('/om2/data/public/imagenet/images_complete/ilsvrc/val', transform=preprocess)
    ds = datasets.ImageNet('ImageNetV2-matched-frequency', # '/om2/data/public/imagenet/images_complete/ilsvrc/',
#     ds = datasets.ImageNet('/om2/data/public/imagenet/images_complete/ilsvrc/',
                           mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                           std=torch.tensor([0.26862954, 0.26130258, 0.27577711]),
                           label_mapping=imagenet2_labelmap,
                           min_value = 0,
                           max_value = 1,
                           aug_train=transforms, # transforms,
                           aug_test=transforms)# transforms)

    ds.scale_image_save_PIL_factor = 255 # Do not scale the output images by 255 when saving with PIL
    ds.init_noise_mean = 0.5
    
    model = AttackerModel(model, ds)

#     from imagenetv2_pytorch import ImageNetV2Dataset
#     images = ImageNetV2Dataset(transform=preprocess)
#     loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=4)

    # send the model to the GPU and return it.
#     model.cuda()
    model.eval()

#     with torch.no_grad():
#         top1, top5, n = 0., 0., 0.
#         for i, (images, target) in enumerate(tqdm(loader)):
# L            images = images.cuda()
 #            target = target.cuda()

#             # predict
#             logits, _ = model(images)
# 
#             # measure accuracy
#             acc1, acc5 = accuracy(logits, target, topk=(1, 5))
#             top1 += acc1
#             top5 += acc5
 #            n += images.size(0)
# 
#     top1 = (top1 / n) * 100
#     top5 = (top5 / n) * 100

#     print(f"Top-1 accuracy: {top1:.2f}")
#     print(f"Top-5 accuracy: {top5:.2f}")

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
