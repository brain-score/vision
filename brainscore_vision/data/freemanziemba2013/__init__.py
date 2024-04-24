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
    version_id=None,  # "Ff3BssTYdRYSXv2hO8ByirYIxsbW1__s",
    sha1="761c08f796db4e342555cdb60eef23a4f19ead43",
    bucket="brainio.contrib",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013.aperture-public'),
)

# assembly: movshon.FreemanZiemba2013.private
data_registry['FreemanZiemba2013.private'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.private",
    version_id=None,  # "3t5ehf.WHJWKX2vUuESVI3PY7ZIUDBkY",
    sha1="63f636fa2e2b51b47a676768a69b06ce95efdd8f",
    bucket="brainio.contrib",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013.aperture-private'),
)

# stimulus set: FreemanZiemba2013.aperture-public
stimulus_set_registry['FreemanZiemba2013.aperture-public'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013.aperture-public",
    bucket="brainio.contrib",
    csv_sha1="194c9b301a4e3e9cff02481d4c19b39331d52660",
    zip_sha1="ad3c6c237491485c863acd2f4d2f219f737e424c",
    csv_version_id="vpWysN7O7LyA9k9PBbrawXauOcqf2LoY",
    zip_version_id="BQmourU.zS4hvTXs.LBIi.yVunyjZNE.",
)

# stimulus set: FreemanZiemba2013.aperture-private
stimulus_set_registry['FreemanZiemba2013.aperture-private'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013.aperture-private",
    bucket="brainio.contrib",
    csv_sha1="65bee31483dd743ae2a19c6af03b9abe5f4d5a41",
    zip_sha1="0015c94e01d037994cdde1b2e3d169ab99f380dc",
    csv_version_id="aNkWLYtttATXQ8GUD8zXIRnWq9mEsdHz",
    zip_version_id="S9eCKFK5b_NcgfnkK2CcddILmkAF8cwP",
)

### secondary assemblies and stimulus sets

# assembly: movshon.FreemanZiemba2013
data_registry['FreemanZiemba2013'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013",
    version_id="knfzP5wWG3BRWgD0PGy1CXCadExElV0f",
    sha1="f03f1630f0ab1e2dbd51816b47fbf2916876134e",
    bucket="brainio.contriib",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013.aperture'),
)

# assembly: movshon.FreemanZiemba2013.noaperture
data_registry['FreemanZiemba2013.noaperture'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.noaperture",
    version_id="3Icsm_HjYeVfzfvsaUJ5qdUT94EaXa5A",
    sha1="6176fd435ab840d284c2c426742c8211622739b5",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013'),
)

# assembly: movshon.FreemanZiemba2013.noaperture.public
data_registry['FreemanZiemba2013.noaperture.public'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.noaperture.public",
    version_id="ggkwDaFLsKesxL0MZY7fMpHuw3gwaEjs",
    sha1="68dd9e1da7207dc2ae17dbbb1adf628e922d15fa",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013-public'),
)

# assembly: movshon.FreemanZiemba2013.noaperture.private
data_registry['FreemanZiemba2013.noaperture.private'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.noaperture.private",
    version_id="SBtg_vxBS_7VsFOy5lBs_.6K_pQnyM40",
    sha1="9ffa772ce61754f6f5b7b391436680ebc25bb8dd",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('FreemanZiemba2013-private'),
)

# stimulus set: FreemanZiemba2013
stimulus_set_registry['FreemanZiemba2013'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013",
    bucket="brainio-brainscore",
    csv_sha1="de0e65a25c7de4c1740f48ac9b1ff513dcfa5caf",
    zip_sha1="bdfc3ba2d878d6aeaa842f9de6abeae50922f2f2",
    csv_version_id="mWuK8gDwe9IC0bdPNPCo1XldIVM_7cdF",
    zip_version_id="IxjtDWqHNtPwByvvSbSueXhrrurY1wjr")

# stimulus set: FreemanZiemba2013.public
stimulus_set_registry['FreemanZiemba2013.public'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013-public",
    bucket="brainio-brainscore",
    csv_sha1="b4fee824f361fa0b78d7247ed4192b04cd675d4f",
    zip_sha1="e14d691db081ace829f76bb24dfc055a4fa2eaf9",
    csv_version_id="js9ggbU.g0bbRLwpbIs6CEQ2JgQIGaRR",
    zip_version_id="nb6oXpiKafGLNK0Ke6z9kEJNdZK4cOjw")

# stimulus set: FreemanZiemba2013.private
stimulus_set_registry['FreemanZiemba2013.private'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013-private",
    bucket="brainio-brainscore",
    csv_sha1="91bab3340410ff5171490a337c5931545f29da82",
    zip_sha1="c2adb4c0f2f0fbbc6006a879234740131ed2dcbb",
    csv_version_id="91GuIJT87Gw5qxv3EfvFD0jf7pmr_XKI",
    zip_version_id="noTN92.53oxESqpZb79dJPl3EjQa41Sd")

# stimulus set: FreemanZiemba2013.aperture
stimulus_set_registry['FreemanZiemba2013.aperture'] = lambda: load_stimulus_set_from_s3(
    identifier="FreemanZiemba2013.aperture",
    bucket="brainio-brainscore",
    csv_sha1="4205eca54974da46accba7812fce25c1b6df58e0",
    zip_sha1="ab07880a0770bd73f68bcd5fd34e6cd945ee17fc",
    csv_version_id="AyDAcEdwVToN_rsCuW2OHj7EZNkOu_we",
    zip_version_id="APXK9Qlgn9PSRCiaWNpBaRla70HW.91h")
