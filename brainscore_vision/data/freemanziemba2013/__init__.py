from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.utils.s3 import load_stimulus_set_from_s3, load_assembly_from_s3
from brainscore_vision.data.data_helpers.helper import version_id_df, build_filename


# assembly: movshon.FreemanZiemba2013.noaperture
data_registry['movshon.FreemanZiemba2013.noaperture'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.noaperture",
    version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013.noaperture', '.nc'), 'version_id'],
    sha1="6176fd435ab840d284c2c426742c8211622739b5",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# assembly: movshon.FreemanZiemba2013.noaperture.public
data_registry['movshon.FreemanZiemba2013.noaperture.public'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.noaperture.public",
    version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013.noaperture.public', '.nc'), 'version_id'],
    sha1="68dd9e1da7207dc2ae17dbbb1adf628e922d15fa",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# assembly: movshon.FreemanZiemba2013.noaperture.private
data_registry['movshon.FreemanZiemba2013.noaperture.private'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.noaperture.private",
    version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013.noaperture.private', '.nc'), 'version_id'],
    sha1="9ffa772ce61754f6f5b7b391436680ebc25bb8dd",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# assembly: movshon.FreemanZiemba2013.public
data_registry['movshon.FreemanZiemba2013.public'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.public",
    version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013.public', '.nc'), 'version_id'],
    sha1="761c08f796db4e342555cdb60eef23a4f19ead43",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# assembly: movshon.FreemanZiemba2013.private
data_registry['movshon.FreemanZiemba2013.private'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013.private",
    version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013.private', '.nc'), 'version_id'],
    sha1="63f636fa2e2b51b47a676768a69b06ce95efdd8f",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# assembly: movshon.FreemanZiemba2013
data_registry['movshon.FreemanZiemba2013'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013",
    version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013', '.nc'), 'version_id'],
    sha1="f03f1630f0ab1e2dbd51816b47fbf2916876134e",
    bucket="brainio.contriib",
    cls=NeuronRecordingAssembly)


# stimulus set: movshon.FreemanZiemba2013
stimulus_set_registry['movshon.FreemanZiemba2013'] = lambda: load_stimulus_set_from_s3(
    identifier="movshon.FreemanZiemba2013",
    bucket="brainio-brainscore",
    csv_sha1="de0e65a25c7de4c1740f48ac9b1ff513dcfa5caf",
    zip_sha1="bdfc3ba2d878d6aeaa842f9de6abeae50922f2f2",
    csv_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013', '.csv'), 'version_id'],
    zip_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013', '.zip'), 'version_id'])

# stimulus set: movshon.FreemanZiemba2013-public
stimulus_set_registry['movshon.FreemanZiemba2013-public'] = lambda: load_stimulus_set_from_s3(
    identifier="movshon.FreemanZiemba2013-public",
    bucket="brainio-brainscore",
    csv_sha1="b4fee824f361fa0b78d7247ed4192b04cd675d4f",
    zip_sha1="e14d691db081ace829f76bb24dfc055a4fa2eaf9",
    csv_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013-public', '.csv'), 'version_id'],
    zip_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013-public', '.zip'), 'version_id'])

# stimulus set: movshon.FreemanZiemba2013-private
stimulus_set_registry['movshon.FreemanZiemba2013-private'] = lambda: load_stimulus_set_from_s3(
    identifier="movshon.FreemanZiemba2013-private",
    bucket="brainio-brainscore",
    csv_sha1="91bab3340410ff5171490a337c5931545f29da82",
    zip_sha1="c2adb4c0f2f0fbbc6006a879234740131ed2dcbb",
    csv_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013-private', '.csv'), 'version_id'],
    zip_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013-private', '.zip'), 'version_id'])

# stimulus set: movshon.FreemanZiemba2013.aperture-public
stimulus_set_registry['movshon.FreemanZiemba2013.aperture-public'] = lambda: load_stimulus_set_from_s3(
    identifier="movshon.FreemanZiemba2013.aperture-public",
    bucket="brainio-brainscore",
    csv_sha1="194c9b301a4e3e9cff02481d4c19b39331d52660",
    zip_sha1="ad3c6c237491485c863acd2f4d2f219f737e424c",
    csv_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013.aperture-public', '.csv'), 'version_id'],
    zip_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013.aperture-public', '.zip'), 'version_id'])

# stimulus set: movshon.FreemanZiemba2013.aperture-private
stimulus_set_registry['movshon.FreemanZiemba2013.aperture-private'] = lambda: load_stimulus_set_from_s3(
    identifier="movshon.FreemanZiemba2013.aperture-private",
    bucket="brainio-brainscore",
    csv_sha1="65bee31483dd743ae2a19c6af03b9abe5f4d5a41",
    zip_sha1="0015c94e01d037994cdde1b2e3d169ab99f380dc",
    csv_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013.aperture-private', '.csv'), 'version_id'],
    zip_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013.aperture-private', '.zip'), 'version_id'])

# stimulus set: movshon.FreemanZiemba2013.aperture
stimulus_set_registry['movshon.FreemanZiemba2013.aperture'] = lambda: load_stimulus_set_from_s3(
    identifier="movshon.FreemanZiemba2013.aperture",
    bucket="brainio-brainscore",
    csv_sha1="4205eca54974da46accba7812fce25c1b6df58e0",
    zip_sha1="ab07880a0770bd73f68bcd5fd34e6cd945ee17fc",
    csv_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013.aperture', '.csv'), 'version_id'],
    zip_version_id=version_id_df.at[build_filename('movshon.FreemanZiemba2013.aperture', '.zip'), 'version_id'])
