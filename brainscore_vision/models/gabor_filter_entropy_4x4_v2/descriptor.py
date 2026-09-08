"""Gabor-filter entropy features for arbitrary image sets.

This is a standalone Python port of the descriptor behind the
GaborFilterEntropy4x4Cosine model from the EntropyReanalysis MATLAB scripts.
The default settings match that model: 256x256 grayscale images, six octave
frequency bands [1, 2, 4, 8, 16, 32], six orientations, 30 degree orientation
bandwidth, entropy in a 4x4 spatial grid, and cosine distance for the RDM.

The key Brain-Score-friendly entry point is ``GaborFilterEntropyExtractor``:
call ``extract_from_paths`` or ``extract_from_images`` to get a
stimulus-by-feature matrix that can later be wrapped as model activations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GaborEntropyConfig:
    """Configurable parameters for the Gabor-filter entropy descriptor."""

    image_size: int = 256
    grid: tuple[int, int] = (4, 4)
    spatial_frequencies: tuple[float, ...] = (1, 2, 4, 8, 16, 32)
    n_orientations: int = 6
    orientation_bandwidth: float = 30.0
    entropy_bins: int = 256
    resize_mode: str = "resize"
    keep_global_entropy: bool = False
    output_dtype: str = "float32"
    valid_extensions: tuple[str, ...] = field(
        default=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    )


class GaborFilterEntropyExtractor:
    """Compute Gabor-filtered block-entropy image descriptors."""

    def __init__(self, config: GaborEntropyConfig | None = None):
        self.config = config or GaborEntropyConfig()
        self._validate_config()
        self.filter_bank = self._make_filter_bank()

    def extract_from_paths(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        """Return an n_images x n_features descriptor matrix."""

        images = [self._load_image(path) for path in image_paths]
        return self.extract_from_images(images)

    def extract_from_images(self, images: Sequence[np.ndarray | Image.Image]) -> np.ndarray:
        """Return descriptors from already-loaded images."""

        features = [self.extract_one(image) for image in images]
        if not features:
            n_filters = len(self.config.spatial_frequencies) * self.config.n_orientations
            n_blocks = self.config.grid[0] * self.config.grid[1]
            n_features = n_filters * n_blocks
            if self.config.keep_global_entropy:
                n_features += n_filters
            return np.zeros((0, n_features), dtype=self.config.output_dtype)
        return np.asarray(features, dtype=self.config.output_dtype)

    def extract_one(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """Return the descriptor for one image."""

        stimulus = self._prepare_image(image)
        spectrum = np.fft.fftshift(np.fft.fft2(stimulus, s=(self.config.image_size, self.config.image_size)))

        block_vectors = []
        global_values = []
        for current_filter in np.moveaxis(self.filter_bank, 2, 0):
            filtered = spectrum * current_filter
            filtered = np.fft.fftshift(filtered)
            filtered = np.fft.ifft2(filtered)
            filtered = np.real(filtered) / (self.config.image_size * self.config.image_size)
            filtered = self._normalize_zero_one(filtered)

            if self.config.keep_global_entropy:
                global_values.append(image_entropy(filtered, self.config.entropy_bins))
            block_vectors.append(self._block_entropy_vector(filtered))

        descriptor = np.concatenate(block_vectors)
        if self.config.keep_global_entropy:
            descriptor = np.concatenate([np.asarray(global_values, dtype=np.float64), descriptor])
        return descriptor.astype(self.config.output_dtype, copy=False)

    def cosine_rdm(self, features: np.ndarray) -> np.ndarray:
        """Return a square cosine-distance RDM from feature rows."""

        return pairwise_cosine_distance(features)

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.image_size <= 0 or cfg.image_size & (cfg.image_size - 1):
            raise ValueError("image_size must be a positive power of two, e.g. 256.")
        if cfg.grid[0] <= 0 or cfg.grid[1] <= 0:
            raise ValueError("grid dimensions must be positive.")
        if cfg.n_orientations <= 0:
            raise ValueError("n_orientations must be positive.")
        if cfg.orientation_bandwidth <= 0:
            raise ValueError("orientation_bandwidth must be positive.")
        if cfg.entropy_bins <= 1:
            raise ValueError("entropy_bins must be greater than one.")
        if cfg.resize_mode not in {"resize", "center_crop", "pad"}:
            raise ValueError("resize_mode must be one of: resize, center_crop, pad.")

    def _load_image(self, path: str | Path) -> Image.Image:
        return Image.open(path).convert("L")

    def _prepare_image(self, image: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(image, Image.Image):
            pil_image = image.convert("L")
        else:
            array = np.asarray(image)
            if array.ndim == 2:
                pil_image = Image.fromarray(to_uint8(array), mode="L")
            elif array.ndim == 3:
                pil_image = Image.fromarray(to_uint8(array)).convert("L")
            else:
                raise ValueError(f"Expected 2D or 3D image array, got shape {array.shape}.")

        pil_image = resize_grayscale_image(pil_image, self.config.image_size, self.config.resize_mode)
        return np.asarray(pil_image, dtype=np.float64) / 255.0

    def _make_filter_bank(self) -> np.ndarray:
        n = self.config.image_size
        n_filters = len(self.config.spatial_frequencies) * self.config.n_orientations
        filter_bank = np.zeros((n, n, n_filters), dtype=np.float64)

        filter_index = 0
        for low_cutoff in self.config.spatial_frequencies:
            high_cutoff = low_cutoff * 2
            frequency_filter = freq_filt(n, low_cutoff, high_cutoff)
            for orientation_index in range(1, self.config.n_orientations + 1):
                orientation = orientation_index * (180 / self.config.n_orientations)
                orientation -= 180 / self.config.n_orientations
                orientation = 180 - orientation
                low_orientation = orientation - self.config.orientation_bandwidth / 2
                high_orientation = orientation + self.config.orientation_bandwidth / 2
                orientation_filter = ori_filt(n, low_orientation, high_orientation)
                filter_bank[:, :, filter_index] = frequency_filter * orientation_filter
                filter_index += 1
        return filter_bank

    def _block_entropy_vector(self, image: np.ndarray) -> np.ndarray:
        row_edges = np.fix(np.linspace(0, image.shape[0], self.config.grid[0] + 1)).astype(int)
        col_edges = np.fix(np.linspace(0, image.shape[1], self.config.grid[1] + 1)).astype(int)
        values = []
        for row_n in range(len(row_edges) - 1):
            row_sel = slice(row_edges[row_n], row_edges[row_n + 1])
            for col_n in range(len(col_edges) - 1):
                col_sel = slice(col_edges[col_n], col_edges[col_n + 1])
                values.append(image_entropy(image[row_sel, col_sel], self.config.entropy_bins))
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _normalize_zero_one(image: np.ndarray) -> np.ndarray:
        image = image - np.min(image)
        image_max = np.max(image)
        if image_max > 0:
            image = image / image_max
        return image


def image_entropy(image: np.ndarray, bins: int = 256) -> float:
    """MATLAB entropy-like Shannon entropy over ``bins`` gray-level bins."""

    clipped = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    quantized = np.floor(clipped * (bins - 1) + 0.5).astype(np.int64)
    counts = np.bincount(quantized.ravel(), minlength=bins).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def pairwise_cosine_distance(features: np.ndarray) -> np.ndarray:
    """Cosine-distance RDM with rows as stimuli and columns as features."""

    features = np.asarray(features, dtype=np.float64)
    norms = np.sqrt(np.sum(features**2, axis=1))
    norms[norms == 0] = np.finfo(np.float64).eps
    similarity = (features @ features.T) / np.outer(norms, norms)
    similarity = np.clip(similarity, -1.0, 1.0)
    rdm = 1.0 - similarity
    rdm = (rdm + rdm.T) / 2.0
    np.fill_diagonal(rdm, 0.0)
    return rdm


def resize_grayscale_image(image: Image.Image, image_size: int, mode: str) -> Image.Image:
    """Return a square grayscale PIL image with side length ``image_size``."""

    if mode == "resize":
        return image.resize((image_size, image_size), Image.BICUBIC)

    width, height = image.size
    if mode == "center_crop":
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize((image_size, image_size), Image.BICUBIC)

    if mode == "pad":
        side = max(width, height)
        canvas = Image.new("L", (side, side), color=128)
        left = (side - width) // 2
        top = (side - height) // 2
        canvas.paste(image, (left, top))
        return canvas.resize((image_size, image_size), Image.BICUBIC)

    raise ValueError(f"Unknown resize mode: {mode}")


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert an image-like array to uint8 without assuming a specific range."""

    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.uint8)
    array = np.where(finite, array, 0)
    if np.min(array) >= 0 and np.max(array) <= 1:
        array = array * 255
    return np.clip(np.round(array), 0, 255).astype(np.uint8)


def discover_images(input_path: str | Path, valid_extensions: Iterable[str]) -> list[Path]:
    """Return sorted image paths from a file, folder, or text file list."""

    path = Path(input_path)
    valid = {extension.lower() for extension in valid_extensions}
    if path.is_dir():
        return sorted(candidate for candidate in path.iterdir() if candidate.suffix.lower() in valid)
    if path.is_file() and path.suffix.lower() in valid:
        return [path]
    if path.is_file():
        base = path.parent
        paths = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            candidate = Path(stripped)
            if not candidate.is_absolute():
                candidate = base / candidate
            paths.append(candidate)
        return paths
    raise FileNotFoundError(f"Could not find image input: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute GaborFilterEntropy grid features and optional cosine RDM."
    )
    parser.add_argument("input", help="Image file, image folder, or text file with one image path per line.")
    parser.add_argument("--output", required=True, help="Output .npz, .csv, or .mat file.")
    parser.add_argument("--save", choices=("features", "rdm", "both"), default="features")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--grid", type=int, nargs=2, default=(4, 4), metavar=("ROWS", "COLS"))
    parser.add_argument("--spatial-frequencies", type=float, nargs="+", default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--n-orientations", type=int, default=6)
    parser.add_argument("--orientation-bandwidth", type=float, default=30.0)
    parser.add_argument("--entropy-bins", type=int, default=256)
    parser.add_argument("--resize-mode", choices=("resize", "center_crop", "pad"), default="resize")
    parser.add_argument("--keep-global-entropy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GaborEntropyConfig(
        image_size=args.image_size,
        grid=tuple(args.grid),
        spatial_frequencies=tuple(args.spatial_frequencies),
        n_orientations=args.n_orientations,
        orientation_bandwidth=args.orientation_bandwidth,
        entropy_bins=args.entropy_bins,
        resize_mode=args.resize_mode,
        keep_global_entropy=args.keep_global_entropy,
    )
    extractor = GaborFilterEntropyExtractor(config)
    image_paths = discover_images(args.input, config.valid_extensions)
    features = extractor.extract_from_paths(image_paths)
    rdm = extractor.cosine_rdm(features) if args.save in {"rdm", "both"} else None
    save_outputs(Path(args.output), args.save, features, rdm, image_paths, config)


def save_outputs(
    output_path: Path,
    save_mode: str,
    features: np.ndarray,
    rdm: np.ndarray | None,
    image_paths: Sequence[Path],
    config: GaborEntropyConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    image_names = np.asarray([path.name for path in image_paths], dtype=object)
    config_dict = {
        "image_size": config.image_size,
        "grid": np.asarray(config.grid),
        "spatial_frequencies": np.asarray(config.spatial_frequencies),
        "n_orientations": config.n_orientations,
        "orientation_bandwidth": config.orientation_bandwidth,
        "entropy_bins": config.entropy_bins,
        "resize_mode": config.resize_mode,
        "keep_global_entropy": config.keep_global_entropy,
    }

    if suffix == ".csv":
        if save_mode == "both":
            raise ValueError("CSV output supports only --save features or --save rdm.")
        np.savetxt(output_path, features if save_mode == "features" else rdm, delimiter=",")
    elif suffix == ".mat":
        from scipy.io import savemat

        payload = {"image_names": image_names, "config": config_dict}
        if save_mode in {"features", "both"}:
            payload["features"] = features
        if save_mode in {"rdm", "both"}:
            payload["rdm"] = rdm
        savemat(output_path, payload)
    elif suffix == ".npz":
        payload = {"image_names": image_names, **config_dict}
        if save_mode in {"features", "both"}:
            payload["features"] = features
        if save_mode in {"rdm", "both"}:
            payload["rdm"] = rdm
        np.savez(output_path, **payload)
    else:
        raise ValueError("Output suffix must be .npz, .csv, or .mat.")


def freq_filt(n: int, low: float, high: float) -> np.ndarray:
    """Port of MATLAB FreqFilt_local(n, low, high, 'G', 'P', 'E', '+')."""

    axis = np.arange(n, dtype=np.float64) - n / 2
    u = np.repeat(axis[:, None], n, axis=1)
    v = np.repeat(axis[None, :], n, axis=0)
    frequency = np.sqrt(u**2 + v**2)
    center = (low + high) / 2
    sc = center - low
    distance = np.abs(center - frequency)

    sc = 1 / np.sqrt(np.log(2)) * sc
    current_filter = np.exp(-((distance / sc) ** 2))
    current_filter[current_filter > 0] = np.sqrt(current_filter[current_filter > 0])
    current_filter[0, 1 : n // 2] = current_filter[0, n // 2 + 1 : n][::-1]
    current_filter[1 : n // 2, 0] = current_filter[n // 2 + 1 : n, 0][::-1]
    return current_filter


def ori_filt(n: int, low: float, high: float) -> np.ndarray:
    """Port of MATLAB OriFilt_local(n, low, high, 'G', 'P', 'E', 'L')."""

    axis = np.arange(n, dtype=np.float64) - n / 2
    u = np.repeat(axis[:, None], n, axis=1)
    v = np.repeat(axis[None, :], n, axis=0)
    radians = np.arctan2(v, u)
    degrees = ((radians * 180) / np.pi) - 90
    degrees[degrees <= -180] += 360

    center = (low + high) / 2
    center_conj = center - 180
    if center_conj <= -180:
        center_conj += 360
    if center_conj > 180:
        center_conj -= 360
    sc = abs(center - low)

    distance_actual = np.abs(center - degrees)
    distance_actual[distance_actual > 180] = 360 - distance_actual[distance_actual > 180]
    distance_conj = np.abs(center_conj - degrees)
    distance_conj[distance_conj > 180] = 360 - distance_conj[distance_conj > 180]
    distance = np.minimum(distance_actual, distance_conj)

    sc = 1 / np.sqrt(np.log(2)) * sc
    current_filter = np.exp(-((distance / sc) ** 2))
    current_filter[current_filter > 0] = np.sqrt(current_filter[current_filter > 0])
    current_filter[0, 1 : n // 2] = current_filter[0, n // 2 + 1 : n][::-1]
    current_filter[1 : n // 2, 0] = current_filter[n // 2 + 1 : n, 0][::-1]
    return current_filter


if __name__ == "__main__":
    main()


class FoveatedGaborFilterEntropyExtractor(GaborFilterEntropyExtractor):
    """Gabor entropy descriptor with smaller central windows and larger peripheral windows."""

    def __init__(
        self,
        config: GaborEntropyConfig | None = None,
        base_stride: int = 24,
        center_window: int = 32,
        mid_window: int = 48,
        peripheral_window: int = 64,
        center_radius: float = 45.0,
        mid_radius: float = 90.0,
    ):
        self.base_stride = base_stride
        self.center_window = center_window
        self.mid_window = mid_window
        self.peripheral_window = peripheral_window
        self.center_radius = center_radius
        self.mid_radius = mid_radius
        super().__init__(config or GaborEntropyConfig())

    def _validate_config(self) -> None:
        super()._validate_config()
        for name in ("base_stride", "center_window", "mid_window", "peripheral_window"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if max(self.center_window, self.mid_window, self.peripheral_window) > self.config.image_size:
            raise ValueError("window sizes cannot exceed image_size.")

    def _block_entropy_vector(self, image: np.ndarray) -> np.ndarray:
        image_size = image.shape[0]
        base_window = self.mid_window
        starts = range(0, image_size - base_window + 1, self.base_stride)
        image_center = (image_size - 1) / 2.0

        values = []
        for row_start in starts:
            base_row_center = row_start + base_window / 2.0
            for col_start in starts:
                base_col_center = col_start + base_window / 2.0
                eccentricity = np.hypot(base_row_center - image_center, base_col_center - image_center)
                window_size = self._window_size_for_eccentricity(eccentricity)
                row_slice, col_slice = self._window_slices(base_row_center, base_col_center, window_size, image_size)
                values.append(image_entropy(image[row_slice, col_slice], self.config.entropy_bins))
        return np.asarray(values, dtype=np.float64)

    def _window_size_for_eccentricity(self, eccentricity: float) -> int:
        if eccentricity <= self.center_radius:
            return self.center_window
        if eccentricity <= self.mid_radius:
            return self.mid_window
        return self.peripheral_window

    @staticmethod
    def _window_slices(row_center: float, col_center: float, window_size: int, image_size: int) -> tuple[slice, slice]:
        row_start = int(round(row_center - window_size / 2.0))
        col_start = int(round(col_center - window_size / 2.0))
        row_start = min(max(row_start, 0), image_size - window_size)
        col_start = min(max(col_start, 0), image_size - window_size)
        return (
            slice(row_start, row_start + window_size),
            slice(col_start, col_start + window_size),
        )
