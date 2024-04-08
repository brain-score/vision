from brainscore_vision.model_helpers.brain_transformation.temporal import time_align, assembly_time_align
import numpy as np

# imports
import numpy as np
from brainio.assemblies import DataAssembly


"""
    This module tests the time alignment functionalities: 
    ie., given a set of target time bins, align the neural assembly with a set of source time bins to them.
"""


T = 3
P = 6
N = 4
time_bins = np.array([(0, 10), (10, 20), (20, 30)])
stimulus_id = ["A", "B", "C", "D", "E", "F"]
neuroid_id = ["n1", "n2", "n3", "n4"]

# latent variable that generates both "video" and "neural signals"
latent1 = [1, 3, 2]
latent2 = [2, 3, 1]
latent3 = [1, 2, 3]
latent4 = [3, 2, 1]
latent5 = [3, 1, 2]
latent6 = [2, 1, 3]
latent = np.array([latent1, latent2, latent3, latent4, latent5, latent6])

# videos
from PIL import Image
vs = np.random.rand(P, T, 8, 8, 3) + latent[..., None, None, None]  # 0~4
vs = (vs * 255//4).astype(np.uint8)
videos = [[Image.fromarray(img) for img in video] for video in vs]
video_paths = [f"video{i}.mp4"for i in range(P)]

# neural signals at time 1 2 3:
n1 = latent * 2
n2 = latent * 1
n3 = latent * .5
n4 = latent * .1
perfect_response = np.array([n1, n2, n3, n4])
noisy_response = perfect_response + np.random.randn(N, P, T) * 0.1

# assemblies
def _make_neural_assembly(data):
    assembly = DataAssembly(
        data,
        dims=["neuroid", "presentation", "time_bin"],
        coords={
            "neuroid_id": ("neuroid", neuroid_id),
            "stimulus_id": ("presentation", stimulus_id),
            "time_bin_start": ("time_bin", time_bins[:, 0]),
            "time_bin_end": ("time_bin", time_bins[:, 1]),
        }
    )
    return assembly

assembly = _make_neural_assembly(noisy_response)

def _except(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except:
        return
    else:
        raise False


def test_time_align():
    # case 1
    source_time_bins = [(0, 100), (100, 200), (200, 300)]
    target_time_bins = [(0, 50), (250, 300)]

    ret = time_align(source_time_bins, target_time_bins, mode = "center")
    belong_to = np.array([[1, 0, 0], [0, 0, 1]])
    assert (ret == belong_to).all()

    ret = time_align(source_time_bins, target_time_bins, mode = "portion")
    belong_to = np.array([[0.5, 0, 0], [0, 0, 0.5]])
    assert (ret == belong_to).all()

    # case 2
    starts = np.arange(0, 100, 10)
    ends = starts + 10
    target_time_bins = np.stack([starts, ends], axis=-1)
    starts = np.arange(0, 100, 1000/30)
    ends = starts + 1000/30
    source_time_bins = np.stack([starts, ends], axis=-1)
    assert (time_align(source_time_bins, target_time_bins).argmax(1) == np.array([0,0,0,1,1,1,1,2,2,2])).all()


def test_assembly_time_align():
    ret = assembly_time_align(assembly, [(0, 10), (20, 30), (10, 20)])
    assert (assembly.isel(time_bin=[0, 2, 1]) == ret.isel(time_bin=[0, 1, 2])).all()
    _except(assembly_time_align, assembly, [(0, 10), (20, 30), (40, 50)])
    _except(assembly_time_align, assembly, [(30, 20)])
    ret = assembly_time_align(assembly, [(20, 30), (0, 10), (10, 20)])
    assert (assembly.isel(time_bin=[2, 0, 1]) == ret.isel(time_bin=[0, 1, 2])).all()

    ret = assembly_time_align(assembly, [(0, 30), (0, 10), (10, 20)], mode="portion")
    assert (assembly.isel(time_bin=[1, 2]) == ret.isel(time_bin=[2, 1])).all()
    assert (ret.isel(time_bin=0).values == assembly.mean("time_bin").values).all()

    ret = assembly_time_align(assembly, [(5, 25)], mode="portion")
    val = (assembly.isel(time_bin=0).values * 0.5 + assembly.isel(time_bin=1).values + assembly.isel(time_bin=2).values * 0.5) / 2
    assert np.isclose(ret.isel(time_bin=0).values, val).all()

    ret = assembly_time_align(assembly, [(9, 21)], mode="portion")
    val = (assembly.isel(time_bin=0).values * .1 + assembly.isel(time_bin=1).values + assembly.isel(time_bin=2).values * .1) / 1.2
    assert np.isclose(ret.isel(time_bin=0).values, val).all()

    ret = assembly_time_align(assembly, [(9, 13)], mode="portion")
    val = (assembly.isel(time_bin=0).values * 0.25 + assembly.isel(time_bin=1).values * 0.75) 
    assert np.isclose(ret.isel(time_bin=0).values, val).all()

    _except(assembly_time_align, assembly, [(20, 35)], mode="portion")
    _except(assembly_time_align, assembly, [(0, 31)], mode="portion")
    _except(assembly_time_align, assembly, [(0, 30.00001)], mode="portion")

    assert (assembly_time_align(assembly, [(0, 10)], mode="portion").values == assembly_time_align(assembly, [(0, 10)], mode="center").values).all()