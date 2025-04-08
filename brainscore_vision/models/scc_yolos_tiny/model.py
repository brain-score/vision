from PIL import Image
from brainscore_vision.model_helpers.check_submission import check_models
import functools
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper

def get_model(name):
    assert name == "yolos_tiny"
    processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    preprocessing = functools.partial(load_preprocess_images, processor=processor)
    wrapper = PytorchWrapper(identifier=name, model=model, preprocessing=preprocessing)
    return wrapper


def get_layers(name):
    assert name == "yolos_tiny"
    return [
        "vit.encoder.layer.0.attention.output.dense",
        "vit.encoder.layer.0.output.dense",
        "vit.encoder.layer.2.attention.output.dense",
        "vit.encoder.layer.2.output.dense",
        "vit.encoder.layer.4.attention.output.dense",
        "vit.encoder.layer.4.output.dense",
        "vit.encoder.layer.6.attention.output.dense",
        "vit.encoder.layer.6.output.dense",
        "vit.encoder.layer.8.attention.output.dense",
        "vit.encoder.layer.8.output.dense",
        "vit.encoder.layer.10.attention.output.dense",
        "vit.encoder.layer.10.output.dense",
        "vit.encoder.layer.11.attention.output.dense",
        "vit.encoder.layer.11.output.dense",

        "vit.layernorm",

        "class_labels_classifier.layers.0",
        "class_labels_classifier.layers.1",
        "class_labels_classifier.layers.2",

        "bbox_predictor.layers.0",
        "bbox_predictor.layers.1",
        "bbox_predictor.layers.2",
    ]
    # layer_names = [
    #     "vit.embeddings.patch_embeddings.projection",
    #     "vit.embeddings.interpolation",
    #     "vit.layernorm",
    #     "class_labels_classifier.layers.2",
    #     "bbox_predictor.layers.2",
    # ]

    # for i in range(12):
    #     layer_names.append(f"vit.encoder.layer.{i}.attention.output.dense") 
    #     layer_names.append(f"vit.encoder.layer.{i}.output.dense")          

    # for n,_ in AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny").named_modules():
    #     print(n)
    return layer_names


def get_bibtex(model_identifier):
    return """
        @article{DBLP:journals/corr/abs-2106-00666,
        author    = {Yuxin Fang and
                    Bencheng Liao and
                    Xinggang Wang and
                    Jiemin Fang and
                    Jiyang Qi and
                    Rui Wu and
                    Jianwei Niu and
                    Wenyu Liu},
        title     = {You Only Look at One Sequence: Rethinking Transformer in Vision through
                    Object Detection},
        journal   = {CoRR},
        volume    = {abs/2106.00666},
        year      = {2021},
        url       = {https://arxiv.org/abs/2106.00666},
        eprinttype = {arXiv},
        eprint    = {2106.00666},
        timestamp = {Fri, 29 Apr 2022 19:49:16 +0200},
        biburl    = {https://dblp.org/rec/journals/corr/abs-2106-00666.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
        }
"""

def load_preprocess_images(image_filepaths, processor=None, **kwargs):
    images = load_images(image_filepaths)
    # Do not resize here — YOLOS processor handles it
    if processor is not None:
        inputs = processor(images=images, return_tensors="pt", **kwargs)
        pixel_values = inputs["pixel_values"]  # [batch, 3, H, W]
        return pixel_values.cpu().numpy()  # For brainscore pipeline
    else:
        raise ValueError("YOLOS requires a processor for preprocessing")


def load_images(image_filepaths):
    return [load_image(image_filepath) for image_filepath in image_filepaths]


def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper() \
                and 'P' not in pil_image.mode.upper():  # not binary and not alpha and not palletized
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image


def preprocess_images(images, image_size, **kwargs):
    preprocess = torchvision_preprocess_input(image_size, **kwargs)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images


def torchvision_preprocess_input(image_size, **kwargs):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        torchvision_preprocess(**kwargs),
    ])


def torchvision_preprocess(normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])

if __name__ == '__main__':
    # get_layers("yolos_tiny")
    check_models.check_base_models(__name__)
