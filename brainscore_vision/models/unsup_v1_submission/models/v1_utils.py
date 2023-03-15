import numpy as np
from PIL import Image


def custom_load_preprocess_images(image_filepaths, image_size, **kwargs):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size, **kwargs)
    return images


def load_images(image_filepaths):
    return [load_image(image_filepath) for image_filepath in image_filepaths]


def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper()\
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


def torchvision_preprocess(core_object=True,normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
    from torchvision import transforms
    if core_object:
        print("Using core_object transform")
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
            lambda img: img.unsqueeze(0)
        ])
    else:
        return transforms.Compose([transforms.Grayscale(),  transforms.ToTensor(),transforms.Normalize(0.47,0.206),
                                   lambda img:img.unsqueeze(0)])


