"""
Method to correct images in the movshon stimulus set by adding a cosine aperture
"""

import argparse
import logging
import os
import numpy as np
import imageio
from tqdm import tqdm
import copy
from pathlib import Path
import pandas as pd
import xarray as xr

from brainio_collection import get_stimulus_set, get_assembly
from brainio_base.stimuli import StimulusSet
from brainio_collection.knownfile import KnownFile as kf
from brainio_contrib.packaging import package_stimulus_set, package_data_assembly
from brainio_collection import fetch

logging.basicConfig(level=logging.DEBUG, filename=f"{__file__}.log", format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


class ApplyCosineAperture:
    def __init__(self, target_dir):
        self._target_dir = target_dir

        self.gray_c = 128
        self.input_degrees = 4
        self.aperture_degrees = 4
        self.pos = np.array([0, 0])
        self.output_degrees = 4
        self.size_px = np.array([320, 320])

        # Image size
        px_deg = self.size_px[0] / self.input_degrees

        self.size_px_out = (self.size_px * (self.output_degrees / self.input_degrees)).astype(int)
        cnt_px = (self.pos * px_deg).astype(int)

        size_px_disp = ((self.size_px_out - self.size_px) / 2).astype(int)

        self.fill_ind = [[(size_px_disp[0] + cnt_px[0]), (size_px_disp[0] + cnt_px[0] + self.size_px[0])],
                        [(size_px_disp[1] + cnt_px[1]), (size_px_disp[1] + cnt_px[1] + self.size_px[1])]]

        # Image aperture
        a = self.aperture_degrees * px_deg / 2
        # Meshgrid with pixel coordinates
        x = (np.arange(self.size_px_out[1]) - self.size_px_out[1] / 2)
        y = (np.arange(self.size_px_out[0]) - self.size_px_out[0] / 2)
        xv, yv = np.meshgrid(x, y)
        # Raised cosine aperture
        inner_mask = (xv - cnt_px[1]) ** 2 + (yv - cnt_px[0]) ** 2 < a ** 2
        cos_mask = 1 / 2 * (1 + np.cos(np.sqrt((xv - cnt_px[1]) ** 2 + (yv - cnt_px[0]) ** 2) / a * np.pi))
        cos_mask[np.logical_not(inner_mask)] = 0

        self.cos_mask = cos_mask

    def convert_image(self, image_path):

        im = imageio.imread(image_path)
        im = im - self.gray_c * np.ones(self.size_px)
        im_template = np.zeros(self.size_px_out)

        im_template[self.fill_ind[0][0]:self.fill_ind[0][1], self.fill_ind[1][0]:self.fill_ind[1][1]] = im
        im_masked = (im_template * self.cos_mask) + self.gray_c * np.ones(self.size_px_out)

        target_path = self._target_dir + os.sep + os.path.basename(image_path)

        imageio.imwrite(target_path, np.uint8(im_masked))

        return target_path


# saves converted image in a new folder given by the target_dir
# returns the converted StimulusSet with the new image_paths and new stimuli_id (with -aperture added in the end)
def convert_stimuli(stimulus_set_existing, stimulus_set_name_new, image_dir_new):
    Path(image_dir_new).mkdir(parents=True, exist_ok=True)

    image_converter = ApplyCosineAperture(target_dir=image_dir_new)
    converted_image_paths = {}
    converted_image_ids = {}
    for image_id in tqdm(stimulus_set_existing['image_id'], total=len(stimulus_set_existing), desc='apply cosine aperture'):
        converted_image_path = image_converter.convert_image(image_path=stimulus_set_existing.get_image(image_id))
        converted_image_id = kf(converted_image_path).sha1
        converted_image_ids[image_id] = converted_image_id
        converted_image_paths[converted_image_id] = converted_image_path
        _logger.debug(f"{image_id} -> {converted_image_id}:  {converted_image_path}")

    converted_stimuli = StimulusSet(stimulus_set_existing.copy(deep=True))
    converted_stimuli["image_id_without_aperture"] = converted_stimuli["image_id"]
    converted_stimuli["image_id"] = converted_stimuli["image_id"].map(converted_image_ids)
    converted_stimuli["image_file_sha1"] = converted_stimuli["image_id"]

    converted_stimuli.image_paths = converted_image_paths
    converted_stimuli.name = stimulus_set_name_new
    converted_stimuli.id_mapping = converted_image_ids

    return converted_stimuli


def update_assembly(assembly, mapping):
    assembly["image_id"] = ("presentation", pd.Series(assembly["image_id"]).map(mapping))
    return assembly


def strip_for_proto(assembly, stimulus_set):
    da = xr.DataArray(assembly).reset_index(assembly.indexes.keys())
    image_level = [k for k, v in da.coords.variables.items() if v.dims == ("presentation",) and
                   k in stimulus_set.columns and k != "image_id"]
    stripped = da.reset_coords(image_level, drop=True)
    for k in list(stripped.attrs):
        del stripped.attrs[k]
    return stripped


def convert_assembly(data_assembly_existing, data_assembly_name_new, stimulus_set_new, mapping):
    stripped = strip_for_proto(data_assembly_existing, stimulus_set_new)
    updated = update_assembly(stripped, mapping)
    updated.name = data_assembly_name_new
    return updated


# main function should be run two times, one for each stimulus set access='public' and access='target'
def main(access):
    local_data_path = fetch._local_data_path
    name_root = 'movshon.FreemanZiemba2013'
    stimulus_set_name_existing = name_root + "-" + access if access != "both" else name_root
    stimulus_set_name_new = name_root + ".aperture-" + access if access != "both" else name_root + ".aperture"
    data_assembly_name_existing = name_root + "." + access if access != "both" else name_root
    data_assembly_name_new = name_root + ".aperture." + access if access != "both" else name_root + ".aperture"
    temp_dir = os.path.join(local_data_path, "temp_" + data_assembly_name_new.replace(".", "_"))

    stimulus_set_existing = get_stimulus_set(stimulus_set_name_existing)
    stimulus_set_new = convert_stimuli(stimulus_set_existing, stimulus_set_name_new, temp_dir)
    mapping = stimulus_set_new.id_mapping
    _logger.debug(f"Packaging stimuli: {stimulus_set_new.name}")
    package_stimulus_set(stimulus_set_new, stimulus_set_name=stimulus_set_new.name,
                         bucket_name="brainio-contrib")

    data_assembly_existing = get_assembly(data_assembly_name_existing)
    proto_data_assembly_new = convert_assembly(data_assembly_existing, data_assembly_name_new, stimulus_set_new, mapping)
    _logger.debug(f"Packaging assembly: {data_assembly_name_new}")
    package_data_assembly(proto_data_assembly_new, data_assembly_name_new, stimulus_set_name_new,
                          bucket_name="brainio-contrib")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Movshon stimuli')
    parser.add_argument('--access', dest='access', type=str,
                      help='access', choices=["both", "public", "private"],
                      default='both')

    args = parser.parse_args()

    main(access=args.access)

