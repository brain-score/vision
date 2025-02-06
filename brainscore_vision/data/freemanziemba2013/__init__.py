from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

BIBTEX = """@article{freeman2013functional,
  title={A functional and perceptual signature of the second visual area in primates},
  author={Freeman, Jeremy and Ziemba, Corey M and Heeger, David J and Simoncelli, Eero P and Movshon, J Anthony},
  journal={Nature neuroscience},
  volume={16},
  number={7},
  pages={974--981},
  year={2013},
  publisher={Nature Publishing Group}
}"""

# assembly: movshon.FreemanZiemba2013.public
data_registry['FreemanZiemba2013.public'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.public",
    version_id="W2vx7jdh9utZdVumHHarXa4L1zOrA8Fd",
    sha1="761c08f796db4e342555cdb60eef23a4f19ead43",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013.aperture-public'),
)

# assembly: movshon.FreemanZiemba2013.private
data_registry['FreemanZiemba2013.private'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.private",
    version_id="wx5LB7Nq42SV74RQ8Cw6EtyabywsnSyG",
    sha1="63f636fa2e2b51b47a676768a69b06ce95efdd8f",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013.aperture-private'),
)

# stimulus set: FreemanZiemba2013.aperture-public
stimulus_set_registry['FreemanZiemba2013.aperture-public'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013.aperture-public",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="194c9b301a4e3e9cff02481d4c19b39331d52660",
    zip_sha1="3a6fec1fc28c212882d6833e267ea8654d289611",
    csv_version_id="null",
    zip_version_id="null",
)

# stimulus set: FreemanZiemba2013.aperture-private
stimulus_set_registry['FreemanZiemba2013.aperture-private'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013.aperture-private",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="65bee31483dd743ae2a19c6af03b9abe5f4d5a41",
    zip_sha1="daa724a4797b23929df59c18744923790edf71cb",
    csv_version_id="Klo0lMygUX0IZPkLoypccXH55a1hZjf2",
    zip_version_id="nZES.el0vF3KXuRrB2ELQOqzCcxumFMY",
)

### secondary assemblies and stimulus sets

# assembly: movshon.FreemanZiemba2013
data_registry['FreemanZiemba2013'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013",
    version_id="null",
    sha1="f03f1630f0ab1e2dbd51816b47fbf2916876134e",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013.aperture'),
)

# assembly: movshon.FreemanZiemba2013.noaperture
data_registry['FreemanZiemba2013.noaperture'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.noaperture",
    version_id="null",
    sha1="6176fd435ab840d284c2c426742c8211622739b5",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013'),
)

# assembly: movshon.FreemanZiemba2013.noaperture.public
data_registry['FreemanZiemba2013.noaperture.public'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.noaperture.public",
    version_id="null",
    sha1="68dd9e1da7207dc2ae17dbbb1adf628e922d15fa",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013.public'),
)

# assembly: movshon.FreemanZiemba2013.noaperture.private
data_registry['FreemanZiemba2013.noaperture.private'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.noaperture.private",
    version_id="null",
    sha1="9ffa772ce61754f6f5b7b391436680ebc25bb8dd",
    bucket="brainscore-storage/brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013.private'),
)

# stimulus set: FreemanZiemba2013
stimulus_set_registry['FreemanZiemba2013'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="de0e65a25c7de4c1740f48ac9b1ff513dcfa5caf",
    zip_sha1="bdfc3ba2d878d6aeaa842f9de6abeae50922f2f2",
    csv_version_id="null",
    zip_version_id="null")

# stimulus set: FreemanZiemba2013.public
stimulus_set_registry['FreemanZiemba2013.public'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013.public",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="b4fee824f361fa0b78d7247ed4192b04cd675d4f",
    zip_sha1="3a5e2cb347eb97d02ff9f7294abc11c1b45a78dc",
    csv_version_id="null",
    zip_version_id="null")

# stimulus set: FreemanZiemba2013.private
stimulus_set_registry['FreemanZiemba2013.private'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013.private",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="91bab3340410ff5171490a337c5931545f29da82",
    zip_sha1="e973cf0f98eac3d9673b00b314deb3b85e76c23c",
    csv_version_id="null",
    zip_version_id="null")

# stimulus set: FreemanZiemba2013.aperture
stimulus_set_registry['FreemanZiemba2013.aperture'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013.aperture",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="4205eca54974da46accba7812fce25c1b6df58e0",
    zip_sha1="3eacddfecd825b18026da3e4c749d5d1bc9213ed",
    csv_version_id="null",
    zip_version_id="null")

