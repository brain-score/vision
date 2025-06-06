from brainio.assemblies import NeuronRecordingAssembly

import brainscore_vision
from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

BIBTEX = """@article{majajhong2015simple,
  title={Simple learned weighted sums of inferior temporal neuronal firing rates accurately predict human core object recognition performance},
  author={Majaj, Najib J and Hong, Ha and Solomon, Ethan A and DiCarlo, James J},
  journal={Journal of Neuroscience},
  volume={35},
  number={39},
  pages={13402--13418},
  year={2015},
  publisher={Soc Neuroscience}
}"""

# assembly: dicarlo.MajajHong2015
data_registry['MajajHong2015'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.MajajHong2015",
    version_id="null",
    sha1="bf8f8d01010d727e3db3f85a9bd5f95f9456b7ec",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('hvm'),
)

# assembly: dicarlo.MajajHong2015.temporal
data_registry['MajajHong2015.temporal'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.MajajHong2015.temporal",
    version_id="null",
    sha1="4c5cfe25ad53162c5c716d64004de269162eff38",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('hvm'),
)

# assembly: dicarlo.MajajHong2015.temporal-10ms
data_registry['MajajHong2015.temporal-10ms'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.MajajHong2015.temporal-10ms",
    version_id="null",
    sha1="3a43db05db722b456d156f53b7215413c994e5b5",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('hvm'),
)

# assembly: dicarlo.MajajHong2015.public
data_registry['MajajHong2015.public'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.MajajHong2015.public",
    version_id="null",
    sha1="13d28ca0ce88ee550b54db3004374ae19096e9b9",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('hvm-public'),
)

# assembly: dicarlo.MajajHong2015.private
data_registry['MajajHong2015.private'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.MajajHong2015.private",
    version_id="null",
    sha1="7a40d16148d6f82939155f0bd976d310857fb156",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('hvm-private'),
)

# assembly: dicarlo.MajajHong2015.temporal.public
data_registry['MajajHong2015.temporal.public'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.MajajHong2015.temporal.public",
    version_id="null",
    sha1="093ac35b3e8464c676d24cf38238415d4d6a9448",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('hvm-public'),
)

# assembly: dicarlo.MajajHong2015.temporal.private
data_registry['MajajHong2015.temporal.private'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.MajajHong2015.temporal.private",
    version_id="null",
    sha1="804ea9e7c08ae9ab7a7d705c9c7e68582750e2ea",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: brainscore_vision.load_stimulus_set('hvm-private'),
)

# stimulus set: hvm - majajhong2015
stimulus_set_registry['hvm'] = lambda: load_stimulus_set_from_s3(
    identifier="hvm",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="a56f55205904d5fb5ead4d0dc7bfad5cc4083b94",
    zip_sha1="6fd5080deccfb061699909ffcb86a26209516811",
    csv_version_id="null",
    zip_version_id="null")

# stimulus set: hvm-public
stimulus_set_registry['hvm-public'] = lambda: load_stimulus_set_from_s3(
    identifier="hvm-public",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="5ca7a3da00d8e9c694a9cd725df5ba0ad6d735af",
    zip_sha1="8aa44e038d7b551efa8077467622f9d48d72e473",
    csv_version_id="null",
    zip_version_id="null")

# stimulus set: hvm-private
stimulus_set_registry['hvm-private'] = lambda: load_stimulus_set_from_s3(
    identifier="hvm-private",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="6ff4981722fa05feb73a2bd26bbbba8b50dc29a6",
    zip_sha1="d7b1ca1876dad87e15b0242b4c82c0203ff3cbd3",
    csv_version_id="null",
    zip_version_id="null")
