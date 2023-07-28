import os
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from pathlib import Path
from pytest import approx

from brainscore_vision.benchmark_helpers.screen import ImageConverter


class TestImageConverter:
    def test_identical(self):
        image_path = Path(os.path.dirname(__file__)) / "rgb1.jpg"
        source_image = np.array(Image.open(image_path))
        with TemporaryDirectory('screen') as target_dir:
            image_converter = ImageConverter(target_dir=target_dir)
            converted_image_path = image_converter.convert_image(image_path, source_degrees=8, target_degrees=8)
            converted_image = np.array(Image.open(converted_image_path))
        assert image_path == converted_image_path
        assert (source_image == converted_image).all()

    def test_crop_shape(self):
        image_path = Path(os.path.dirname(__file__)) / "rgb1.jpg"
        with TemporaryDirectory('screen') as target_dir:
            image_converter = ImageConverter(target_dir=target_dir)
            converted_image_path = image_converter.convert_image(image_path, source_degrees=10, target_degrees=8)
            source_image, converted_image = Image.open(image_path), Image.open(converted_image_path)
        np.testing.assert_array_equal(converted_image.size, [1855 * .8, 1855 * .8])

    def test_enlarge_shape(self):
        image_path = Path(os.path.dirname(__file__)) / "rgb1.jpg"
        with TemporaryDirectory('screen') as target_dir:
            image_converter = ImageConverter(target_dir=target_dir)
            converted_image_path = image_converter.convert_image(image_path, source_degrees=10, target_degrees=12)
            source_image, converted_image = Image.open(image_path), Image.open(converted_image_path)
        np.testing.assert_array_equal(converted_image.size, [1855 * 1.2, 1855 * 1.2])

    def test_enlarge_match(self):
        image_path = Path(os.path.dirname(__file__)) / "rgb1.jpg"
        with TemporaryDirectory('screen') as target_dir:
            image_converter = ImageConverter(target_dir=target_dir)
            converted_image_path = image_converter.convert_image(image_path, source_degrees=10, target_degrees=12)
            converted_image = Image.open(converted_image_path)
        target_image = Image.open(Path(os.path.dirname(__file__)) / "rgb1-10to12.jpg")
        np.testing.assert_array_equal(np.array(converted_image), np.array(target_image))

    def test_enlarge_gray_background(self):
        image_path = Path(os.path.dirname(__file__)) / "rgb1.jpg"
        with TemporaryDirectory('screen') as target_dir:
            image_converter = ImageConverter(target_dir=target_dir)
            converted_image_path = image_converter.convert_image(image_path, source_degrees=8, target_degrees=10)
            converted_image = Image.open(converted_image_path)

        converted_image = np.array(converted_image)
        amount_gray = 0
        for index in np.ndindex(converted_image.shape[:2]):
            color = converted_image[index]
            gray = [128, 128, 128]
            if (color == gray).all():
                amount_gray += 1
        assert amount_gray / converted_image.size == approx(.11658, abs=.0001)
        assert amount_gray == 1880753

    def test_enlarge_image_centered(self):
        image_path = Path(os.path.dirname(__file__)) / "rgb1.jpg"
        with TemporaryDirectory('screen') as target_dir:
            image_converter = ImageConverter(target_dir=target_dir)
            converted_image_path = image_converter.convert_image(image_path, source_degrees=8, target_degrees=10)
            converted_image = Image.open(converted_image_path)

        converted_image = np.array(converted_image)
        gray = [128, 128, 128]
        assert (converted_image[48, 48] == gray).all()
        assert (converted_image[224 - 48, 224 - 48] == gray).all()
        assert (converted_image[48, 224 - 48] == gray).all()
        assert (converted_image[224 - 48, 48] == gray).all()
