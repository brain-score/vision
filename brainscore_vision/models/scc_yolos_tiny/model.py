from PIL import Image
from brainscore_vision.model_helpers.check_submission import check_models
import functools
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images

def get_model(name):
    assert name == "yolos_tiny"
    
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")

    def preprocess_yolos(image_paths):
        # Load images from disk
        images = [Image.open(path).convert("RGB") for path in image_paths]
        # Use Hugging Face processor to do padding, resizing, tensor conversion
        inputs = processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]
        
    wrapper = PytorchWrapper(identifier='yolos_tiny', model=model, preprocessing=preprocess_yolos, batch_size=4)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == "yolos_tiny"
    layer_names = [
        "vit.embeddings.patch_embeddings.projection",
        "vit.embeddings.interpolation",
        "vit.layernorm",
        "class_labels_classifier.layers.2",
        "bbox_predictor.layers.2",
    ]

    for i in range(12):
        layer_names.append(f"vit.encoder.layer.{i}.attention.output.dense") 
        layer_names.append(f"vit.encoder.layer.{i}.output.dense")          

    # for n,_ in AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny").named_modules():
    #     print(n)
    return layer_names


def get_bibitex(model_identifier):
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

if __name__ == '__main__':
    # get_layers("yolos_tiny")
    check_models.check_base_models(__name__)
