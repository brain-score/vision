from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3
from brainscore_vision.data_helpers.lookup_legacy import version_id_df, build_filename

BIBTEX = """@article{kuzovkin2018activations,
  title={Activations of deep convolutional neural networks are aligned with gamma band activity of human visual cortex},
  author={Kuzovkin, Ilya and Vicente, Raul and Petton, Mathilde and Lachaux, Jean-Philippe and Baciu, Monica and Kahane, Philippe and Rheims, Sylvain and Vidal, Juan R and Aru, Jaan},
  journal={Communications biology},
  volume={1},
  number={1},
  pages={1--12},
  year={2018},
  publisher={Nature Publishing Group}
}"""

# extract version ids from version_ids csv
assembly_version = version_id_df.at[build_filename('aru.Kuzovkin2018', '.nc'), 'version_id']
csv_version = version_id_df.at[build_filename('aru.Kuzovkin2018', '.csv'), 'version_id']
zip_version = version_id_df.at[build_filename('aru.Kuzovkin2018', '.zip'), 'version_id']

# assembly
data_registry['aru.Kuzovkin2018'] = lambda: load_assembly_from_s3(
    identifier="aru.Kuzovkin2018",
    version_id=assembly_version,
    sha1="5fae8b283a043562ce9925d48ad99db151f39067",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly)

# stimulus set
stimulus_set_registry['aru.Kuzovkin2018'] = lambda: load_stimulus_set_from_s3(
    identifier="aru.Kuzovkin2018",
    bucket="brainio-brainscore",
    csv_sha1="a5990b24aea3e453756141cbe69a83304db72d0b",
    zip_sha1="cca4d819d7743bdd4bf65c1cb2439fd0ec97543a",
    csv_version_id=csv_version,
    zip_version_id=zip_version)
