from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.utils.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_vision.data.data_helpers.helper import version_id_df, build_filename


BIBTEX = """"""


# public assembly: uses dicarlo.objectome.public stimuli
data_registry['dicarlo.Rajalingham2018.public'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rajalingham2018.public",
    version_id=version_id_df.at[build_filename('dicarlo.Rajalingham2018.public', '.nc'), 'version_id'],
    sha1="34c6a8b6f7c523589c1861e4123232e5f7c7df4c",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)

# private assembly: uses dicarlo.objectome.private stimuli
data_registry['dicarlo.Rajalingham2018.private'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rajalingham2018.private",
    version_id=version_id_df.at[build_filename('dicarlo.Rajalingham2018.private', '.nc'), 'version_id'],
    sha1="516f13793d1c5b72bb445bb4008448ce97a02d23",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly)


# stimulus set: dicarlo.objectome.public - rajalingham2018
stimulus_set_registry['dicarlo.objectome.public'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.objectome.public",
    bucket="brainio-brainscore",
    csv_sha1="47884e17106a3be471d6481279cab33889b80850",
    zip_sha1="064f2955f98e63867755fee2e3ead8cddf6bfab8",
    csv_version_id=version_id_df.at[build_filename('dicarlo.objectome.public', '.csv'), 'version_id'],
    zip_version_id=version_id_df.at[build_filename('dicarlo.objectome.public', '.zip'), 'version_id'])

# stimulus set: dicarlo.objectome.private - same
stimulus_set_registry['dicarlo.objectome.private'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.objectome.private",
    bucket="brainio-brainscore",
    csv_sha1="ac38e8f7c08fa8294ed25a3bf84a6adb108bf3fc",
    zip_sha1="ccd39f7f9b8b4a92da06e3960d06225e46208593",
    csv_version_id=version_id_df.at[build_filename('dicarlo.objectome.private', '.csv'), 'version_id'],
    zip_version_id=version_id_df.at[build_filename('dicarlo.objectome.private', '.zip'), 'version_id'])
