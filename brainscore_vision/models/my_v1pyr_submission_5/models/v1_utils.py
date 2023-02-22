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



def interpolate1d(x_new, Y, X):
    r""" One-dimensional linear interpolation.

    Returns the one-dimensional piecewise linear interpolant to a
    function with given discrete data points (X, Y), evaluated at x_new.

    Note: this function is just a wrapper around ``np.interp()``.

    Parameters
    ----------
    x_new: torch.Tensor
        The x-coordinates at which to evaluate the interpolated values.
    Y: array_like
        The y-coordinates of the data points.
    X: array_like
        The x-coordinates of the data points, same length as X.

    Returns
    -------
    Interpolated values of shape identical to `x_new`.

    TODO
    ----
    rename and reorder arguments, refactor corresponding use in SteerablePyr
    """

    out = np.interp(x=x_new.flatten(), xp=X, fp=Y)

    return np.reshape(out, x_new.shape)


def raised_cosine(width=1, position=0, values=(0, 1)):
    """Return a lookup table containing a "raised cosine" soft threshold
    function

    Y =  VALUES(1)
        + (VALUES(2)-VALUES(1))
        * cos^2( PI/2 * (X - POSITION + WIDTH)/WIDTH )

    this lookup table as suitable for use by `interpolate1d`

    Parameters
    ---------
    width : float
        the width of the region over which the transition occurs
    position : float
        the location of the center of the threshold
    values : tuple
        2-tuple specifying the values to the left and right of the transition.

    Returns
    -------
    X : `np.ndarray`
        the x values of this raised cosine
    Y : `np.ndarray`
        the y values of this raised cosine
    """

    sz = 256   # arbitrary!

    X = np.pi * np.arange(-sz-1, 2) / (2*sz)

    Y = values[0] + (values[1]-values[0]) * np.cos(X) ** 2

    # make sure end values are repeated, for extrapolation...
    Y[0] = Y[1]
    Y[sz+2] = Y[sz+1]

    X = position + (2*width/np.pi) * (X + np.pi / 4)

    return X, Y