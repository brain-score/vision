import pytest
# from utils import get_clip_vision_model
import importlib
utils = importlib.import_module('sample-model-submission.utils')
get_clip_vision_model = utils.get_clip_vision_model

def test_clip_load():
    print(get_clip_vision_model('RN50'))
    print(get_clip_vision_model('RN50x4'))
    print(get_clip_vision_model('RN50x16'))
    print(get_clip_vision_model('RN101'))