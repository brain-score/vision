import numpy as np
from PIL import Image
from pathlib import Path
from pytest import approx
from tempfile import TemporaryDirectory

from brainscore_vision.benchmark_helpers.screen import ImageConverter


class TestImageConverter:
    def test_identical(self):
        image_path = Path(__file__).parent / "rgb1.jpg"
        source_image = np.array(Image.open(image_path))
        with TemporaryDirectory('screen') as target_dir:
            image_converter = ImageConverter(target_dir=target_dir)
            converted_image_path = image_converter.convert_image(image_path, source_degrees=8, target_degrees=8)
            converted_image = np.array(Image.open(converted_image_path))
        assert image_path == converted_image_path
        assert (source_image == converted_image).all()

    def test_crop_shape(self):
        image_path = Path(__file__).parent / "rgb1.jpg"
        with TemporaryDirectory('screen') as target_dir:
            image_converter = ImageConverter(target_dir=target_dir)
            converted_image_path = image_converter.convert_image(image_path, source_degrees=10, target_degrees=8)
            source_image, converted_image = Image.open(image_path), Image.open(converted_image_path)
        np.testing.assert_array_equal(converted_image.size, [1855 * .8, 1855 * .8])

    def test_enlarge_shape(self):
        image_path = Path(__file__).parent / "rgb1.jpg"
        with TemporaryDirectory('screen') as target_dir:
            image_converter = ImageConverter(target_dir=target_dir)
            converted_image_path = image_converter.convert_image(image_path, source_degrees=10, target_degrees=12)
            source_image, converted_image = Image.open(image_path), Image.open(converted_image_path)
        np.testing.assert_array_equal(converted_image.size, [1855 * 1.2, 1855 * 1.2])

    def test_enlarge_match(self):
        image_path = Path(__file__).parent / "rgb1.jpg"
        with TemporaryDirectory('screen') as target_dir:
            image_converter = ImageConverter(target_dir=target_dir)
            converted_image_path = image_converter.convert_image(image_path, source_degrees=10, target_degrees=12)
            converted_image = Image.open(converted_image_path)
        target_image = Image.open(Path(__file__).parent / "rgb1-10to12.png")
        np.testing.assert_array_equal(converted_image, target_image)

    def test_enlarge_gray_background(self):
        image_path = Path(__file__).parent / "rgb1.jpg"
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
        assert amount_gray == approx(1_880_000, rel=.0005)  # depending on Pillow version for some reason

    def test_enlarge_image_centered(self):
        image_path = Path(__file__).parent / "rgb1.jpg"
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


def _table(tmp_path, count=40):
    import pandas as pd
    from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
    sources = tmp_path / 'sources'
    sources.mkdir(exist_ok=True)
    for index in range(count):
        Image.new('RGB', (96, 96), (index * 6, 90, 200)).save(sources / f'image{index}.png')
    ids = [f'image{index}' for index in range(count)]
    table = StimulusSet(pd.DataFrame({'stimulus_id': ids}))
    table.stimulus_paths = {stimulus_id: str(sources / f'{stimulus_id}.png') for stimulus_id in ids}
    table.identifier = 'two-writers'
    return table


def _convert(table):
    """Names and bytes of every converted image (the store decorator is skipped)."""
    from brainscore_vision.benchmark_helpers import screen
    converted = screen._place_on_screen.__wrapped__(
        stimuli_identifier=table.identifier, stimulus_set=table,
        target_visual_degrees=8, source_visual_degrees=10)
    paths = {i: Path(converted.get_stimulus(i)) for i in table['stimulus_id']}
    return {i: (path.name, path.read_bytes()) for i, path in paths.items()}


def _alone(tmp_path, monkeypatch, table):
    from brainscore_vision.benchmark_helpers import screen
    monkeypatch.setattr(screen, 'root_path', tmp_path / 'alone')
    result = _convert(table)
    monkeypatch.setattr(screen, 'root_path', tmp_path / 'screen')
    return result


def test_a_run_that_finds_a_half_written_screen_folder_gives_the_images_of_a_run_alone(tmp_path, monkeypatch):
    """A folder that an older run left half written is replaced, not used."""
    table = _table(tmp_path)
    alone = _alone(tmp_path, monkeypatch, table)
    half = tmp_path / 'screen' / 'two-writers--target8.00--source10.00'
    half.mkdir(parents=True)
    (half / 'image0.png').write_bytes(b'half')
    assert _convert(table) == alone


def test_a_screen_folder_with_every_name_and_other_bytes_gives_the_new_bytes(tmp_path, monkeypatch):
    """A folder from an older conversion (other bytes under the same names) is replaced, not used."""
    table = _table(tmp_path, count=3)
    alone = _alone(tmp_path, monkeypatch, table)
    old = tmp_path / 'screen' / 'two-writers--target8.00--source10.00'
    old.mkdir(parents=True)
    for name, _ in alone.values():
        (old / name).write_bytes(b'older conversion')
    assert _convert(table) == alone
    assert [path.name for path in (tmp_path / 'screen').iterdir()] == ['two-writers--target8.00--source10.00']


def test_a_screen_folder_path_that_is_a_file_gives_an_error_and_no_paths(tmp_path, monkeypatch):
    import pytest
    table = _table(tmp_path, count=3)
    _alone(tmp_path, monkeypatch, table)
    (tmp_path / 'screen').mkdir()
    (tmp_path / 'screen' / 'two-writers--target8.00--source10.00').write_text('not a folder')
    from brainscore_vision.benchmark_helpers import screen
    with pytest.raises(OSError, match='two-writers--target8.00--source10.00'):  # not paths to missing images
        screen._place_on_screen.__wrapped__(stimuli_identifier=table.identifier, stimulus_set=table,
                                            target_visual_degrees=8, source_visual_degrees=10)
    assert [path.name for path in (tmp_path / 'screen').iterdir()] == ['two-writers--target8.00--source10.00']


def test_a_conversion_that_fails_leaves_no_work_folder(tmp_path, monkeypatch):
    import pytest
    from brainscore_vision.benchmark_helpers import screen
    table = _table(tmp_path, count=3)
    table.stimulus_paths['image2'] = str(tmp_path / 'missing.png')
    monkeypatch.setattr(screen, 'root_path', tmp_path / 'screen')
    with pytest.raises(OSError):
        _convert(table)
    assert list((tmp_path / 'screen').iterdir()) == []


def test_two_writers_of_one_screen_folder_both_succeed_with_equal_images(tmp_path, monkeypatch):
    """Two runs on one store convert the same table at once (the cloud runs of two models on one
    benchmark): both succeed, and every image they name has the name and bytes of a run alone."""
    import threading
    table = _table(tmp_path)
    alone = _alone(tmp_path, monkeypatch, table)
    for _ in range(5):
        start, results, errors = threading.Barrier(2), [], []

        def convert():
            start.wait()
            try:
                results.append(_convert(table))
            except Exception as error:
                errors.append(error)
        writers = [threading.Thread(target=convert) for _ in range(2)]
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join()
        assert not errors and results == [alone, alone]
    assert [path.name for path in (tmp_path / 'screen').iterdir()] == ['two-writers--target8.00--source10.00']
