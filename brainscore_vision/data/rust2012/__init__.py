from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3

BIBTEX = """@article{rust2012balanced,
  title={Balanced increases in selectivity and tolerance produce constant sparseness along the ventral visual stream},
  author={Rust, Nicole C and DiCarlo, James J},
  journal={Journal of Neuroscience},
  volume={32},
  number={30},
  pages={10170--10182},
  year={2012},
  publisher={Soc Neuroscience}
}"""

# dicarlo.Rust2012.single assembly
data_registry['dicarlo.Rust2012.single'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rust2012.single",
    version_id="8T7Pt1LwHPRoxCC9c9xBfMNGDSJ0b3e4",
    sha1="4ef420e70fbd0de3745df5be7c83dfc0a8f2e528",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('dicarlo.Rust2012'),
)

# dicarlo.Rust2012.array assembly
data_registry['dicarlo.Rust2012.array'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rust2012.array",
    version_id="O_8qaKqFhzRNOIvVx8djTQDUE0a6g.w9",
    sha1="6709b641751370acfccd9567e3d75b71865a71ab",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('dicarlo.Rust2012'),
)


# stimulus set
stimulus_set_registry['dicarlo.Rust2012'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Rust2012",
    bucket="brainio-brainscore",
    csv_sha1="482da1f9f4a0ab5433c3b7b57073ad30e45c2bf1",
    zip_sha1="7cbf5dcec235f7705eaad1cfae202eda77e261a2",
    csv_version_id="wtjzAmRyAgzZUHcn_ThuAnlIbr5W7o3y",
    zip_version_id=".lwNwrP6vLz3cnOOb6eqf8atHSOKoL6X")
