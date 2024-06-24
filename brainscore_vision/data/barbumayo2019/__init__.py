from brainscore_vision import stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3

BIBTEX = """@inproceedings{NEURIPS2019_97af07a1,
 author = {Barbu, Andrei and Mayo, David and Alverio, Julian and Luo, William and Wang, Christopher and Gutfreund, Dan and Tenenbaum, Josh and Katz, Boris},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models},
 url = {https://proceedings.neurips.cc/paper/2019/file/97af07a14cacba681feacf3012730892-Paper.pdf},
 volume = {32},
 year = {2019}
}"""

# stimulus set
stimulus_set_registry['BarbuMayo2019'] = lambda: load_stimulus_set_from_s3(
    identifier="BarbuMayo2019",
    bucket="brainio-brainscore",
    csv_sha1="e4d8888ccb6beca28636e6698e7beb130e278e12",
    zip_sha1="1365eb2a7231516806127a7d2a908343a7ac9464",
    csv_version_id="3H1nicJcmKZgqfPV.ouH5jZln3GlZ3Lr",
    zip_version_id="sx9cN3NFBBnQCnUCa85ziyWJxTUaPJGf")
