from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

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


# assembly
data_registry['Kuzovkin2018'] = lambda: load_assembly_from_s3(
    identifier="aru.Kuzovkin2018",
    version_id="nk5w.m3N1D4fWg2PeY9_AJb5yY6UtPeM",
    sha1="c3787b30ffdc6816e245a8f8d8f2096d8bf85569",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kuzovkin2018'),
)

# stimulus set
stimulus_set_registry['Kuzovkin2018'] = lambda: load_stimulus_set_from_s3(
    identifier="aru.Kuzovkin2018",
    bucket="brainio-brainscore",
    csv_version_id="XcZTPe65cRJ_vMibAAQOpKyXin0MDWnG",
    csv_sha1="f40cbe9f385c2f391952bcc91cec3878a18f0bc4",
    zip_version_id="zg5gixyw1mI3e.45CH.O81v8Jqb3dgJp",
    zip_sha1="9698c22f30b19d68bbb72195eaefbdf4a5899f57",
)
