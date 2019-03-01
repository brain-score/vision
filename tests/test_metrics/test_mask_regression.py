import os

import numpy as np
import xarray as xr
from brainio_base.stimuli import StimulusSet
from keras.applications.vgg19 import VGG19
from model_tools.activations.keras import KerasWrapper, preprocess
from pytest import approx

from brainio_base.assemblies import NeuronRecordingAssembly
from brainscore.metrics.mask_regression import CubeMapper


class TestCubeMapper:
    def test_mapper(self):
        model_name = 'vgg-19'
        layer = 'block2_pool'
        test_nc = 'ones_testing.nc'
        im_dir = 'images'

        load_preprocess = lambda image_filepaths: preprocess(image_filepaths, image_size=224)

        params = {
            "_model_name": model_name
            , '_layer': layer
            , '_assembly': load_nc_data()
            , '_im_dir': im_dir
            , '_nc_file': test_nc
            , '_wrapper': KerasWrapper(VGG19(), load_preprocess)
        }
        #
        # default {ls: 0.05 , ld: 0.1, lr: 0.01}
        num_neurons = 18
        block2_pool_output_shape = 56 * 56
        block2_pool_num_filter = 128
        # input shape:
        inits = {
            's_w': np.ones((num_neurons, block2_pool_output_shape)),
            's_d': np.ones((num_neurons, block2_pool_num_filter))
        }

        mapper_params = {
            "num_neurons": num_neurons
            , "max_epochs": 1
            , "log_rate": 10
            , "gpu_options": None
            , "inits": inits
        }
        #
        check = CubeMapper(**params)
        check.init_mapper(**mapper_params)
        #
        score = check()

        assert isinstance(score, float)
        assert score == approx(470.09, abs=0.1)



def load_nc_data():
    nc_load = xr.open_dataarray(os.path.join(os.path.dirname(__file__), 'ones-mask_testing.nc'))
    assembly = NeuronRecordingAssembly(nc_load)
    assembly = assembly.squeeze("time_bin")
    assembly = assembly.transpose('presentation', 'neuroid')
    # fill nan with response mean
    assembly = assembly.fillna(assembly.mean(dim=('neuroid'), skipna=True))
    return assembly
