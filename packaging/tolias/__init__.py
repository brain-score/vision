import numpy as np
from PIL import Image
import os


def main():
    data_path = os.path.join(os.path.dirname(__file__), 'cadena2017')
    images = np.load(os.path.join(data_path, 'images.npy'))
    _show_image(images[0])
    np.testing.assert_array_equal(images.shape, (7250, 140, 140))  # images x width x height
    responses = np.load(os.path.join(data_path, 'responses.npy'))
    np.testing.assert_array_equal(responses.shape, (4, 7250, 166))  # num_repetitions x num_images x num_neurons
    print("{}/{} responses NaN ({:.2f}%)".format(
        np.isnan(responses).sum(), responses.size, np.isnan(responses).sum() / responses.size * 100))


def _show_image(img, savepath=None):
    img = Image.fromarray((img * 255).astype('uint8'))
    if savepath:
        img.save(savepath)
    else:
        img.show()


if __name__ == '__main__':
    main()
