from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_core.supported_data_standards.brainio.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = """@article{Kato2026,
    title = {Systematic image perturbations reveal persistent gaps between human and machine vision},
    volume = {XX},
    doi = {XX},
    journal = {iScience},
    author = {Kato, Mugihiko, and He, Biyu},
    year = {2026},
    }"""
    
# stimulus set
stimulus_set_registry['Kato2026.Ori'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Ori',
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    csv_sha1="8e3eee196d5b1bca058c8dad54234fa5340eeaca",
    zip_sha1="75b5720f107eadaa9689e80bf812a173f1f099e9",
    csv_version_id="v37pD8U49wkFeTYQ7sLt397EyC2esf.v",
    zip_version_id="F9GXCJblKO.PdKLn_NcDemmxmIztTGIf")

# assembly
data_registry['Kato2026.Ori'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Ori',
    version_id="nutKX32TrJlXRU5gmELvhuAbpjvP3MS6",
    sha1="d5a22c25835da413b629d87ec77d6af834a43732",
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Ori'),
)
                
# stimulus set
stimulus_set_registry['Kato2026.Fil'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Fil',
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    csv_sha1="469b6af5259450cda4eec7586ccbbcc5174aa0c4",
    zip_sha1="9b7a45311b1caa9b2f16bc97bc4ee24eb197265b",
    csv_version_id="0gB3zLHFu3sKxHyjU3IV3V2oVIrkevSq",
    zip_version_id="nfelaldfjl1xhtlICZj1gJrRTZTT63I5")

# assembly
data_registry['Kato2026.Fil'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Fil',
    version_id="y8TPPNoBeb_gpFiRpRzOmZzzjbRbawSL",
    sha1="954498ac9cd106fb17443bc77ac2c00d1fe03f00",
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Fil'),
)

# stimulus set
stimulus_set_registry['Kato2026.Lin'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Lin',
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    csv_sha1="36b8045876aa47151a65f16ddee645cf9473f4fb",
    zip_sha1="2b786b9901a05d1fd980ae34153536cf906a8aa1",
    csv_version_id="IL56F4.Rz9.Xzy2oUhNsHBUut.8s3WF0",
    zip_version_id="8yGnuNsJd4GgZhq5HHkrEmy5mJDTWCHk")

# assembly
data_registry['Kato2026.Lin'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Lin',
    version_id="NjP.6gFeujPu8ZIkO_zzLnxbPGfTVIfg",
    sha1="ce681e8afca2d39166ae2f19ba665638f794b9d1",
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Lin'),
)

# stimulus set
stimulus_set_registry['Kato2026.SilTex'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.SilTex',
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    csv_sha1="ff982583d8d815247cc1a009255857772e07cd94",
    zip_sha1="6cdc21439438e126f3ab401bb05af84300a4fe1e",
    csv_version_id="HSe4.AKWoZC9k7kS_bN0AeCtfjfdTJB2",
    zip_version_id="ZQAuxXQl85ZPpzndP20u5PY8GAIH_TLx")

# assembly
data_registry['Kato2026.SilTex'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.SilTex',
    version_id="hJ4wbDVpd7MZ.lEFDpS9F7DIqUj.OkGL",
    sha1="ce1474dea31dd0b7469ec634e5e61da55e414502",
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.SilTex'),
)

# stimulus set
stimulus_set_registry['Kato2026.Sil'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Sil',
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    csv_sha1="13a42a068aa79fd533fd7b5f00a09f67053b4425",
    zip_sha1="0d02bea9d282d020d9b20ceb7daa50d185cdefe3",
    csv_version_id="MqUxdUlvPKXj7JD7EJIMxNBRKvaME7yr",
    zip_version_id="7ywqzWvPByIcPNvpu8khwJgdZcXriABk")

# assembly
data_registry['Kato2026.Sil'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Sil',
    version_id="B2ijyQPmqtqVWcOolcDx.J1zI5u1BTHl",
    sha1="4a21ce43ca5f7dc0b8d9c3d048ea2dda2ca7f47a",
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Sil'),
)

# stimulus set
stimulus_set_registry['Kato2026.Tex'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Tex',
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    csv_sha1="cd533001022f56bef5b39ce43b00a4a4293a81d1",
    zip_sha1="fb6564cae35ca18172025b46cd653f4721015b3d",
    csv_version_id="YkFB1GygdahN8vmQoYgEdCbFuMopCLDE",
    zip_version_id="4URogRZy2_VicJSZtt76RGUlekY9KzrH")

# assembly
data_registry['Kato2026.Tex'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Tex',
    version_id="2OgRnOhsF2mCGfRFFf9RkiLqUkXB4m7m",
    sha1="cb4d428412d4334f57e9b7b14c7e7d888bf838db",
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Tex'),
)

# stimulus set
stimulus_set_registry['Kato2026.LinFil'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.LinFil',
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    csv_sha1="9fbeb46e9a2a5ac0573ebf4ce6bb20fb9ddf3a33",
    zip_sha1="2ac934e65c3645b960c3e1126921cea783da6479",
    csv_version_id="Hf71APG6P8vuJQcW6N3WuEWe_tUlNaXY",
    zip_version_id="UBqoMpF9jKUR5MOBMxxRrMsMPnaE8NKn")

# assembly
data_registry['Kato2026.LinFil'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.LinFil',
    version_id="pn_0hSTxJSsw.ief2tpEG6rktfZ47zVd",
    sha1="87f0227c3507e8cf1172ddf5398036f098a40c96",
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.LinFil'),
)

# stimulus set
stimulus_set_registry['Kato2026.Sparse'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Sparse',
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    csv_sha1="990d317b8d526590257d4acc0abf70e89a4f6661",
    zip_sha1="67569fca4b89b9ae4e893f4eeb71e3b2c2c67808",
    csv_version_id="_flud7h8c_r2mtgCK1dQ2zbjWbHwHhxU",
    zip_version_id="dO3QpHlVFduyYl6YlY84PL2g7cSpgByz")

# assembly
data_registry['Kato2026.Sparse'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Sparse',
    version_id="WhBrpC7mvHBirLmNtUn3cBPMm_9e_UmL",
    sha1="e933f23d2c09a2107b292657063a99d0a931119e",
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Sparse'),
)

# stimulus set
stimulus_set_registry['Kato2026.Dense'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Dense',
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    csv_sha1="4e1b9387371d416a0c5758cd78ee683c23c7e191",
    zip_sha1="7557f8e1542cd14edd1f4cf79264ea4851861651",
    csv_version_id="yj0tgN6gKBP1n90NvJNL7ouMyidQ8ObR",
    zip_version_id="k1_PHVYLBdTY6oibqdn5yIuR1gs9MJRE")

# assembly
data_registry['Kato2026.Dense'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Dense',
    version_id="noe8YWgx282I3FDCCtydNX2f.l0f2Luu",
    sha1="49b2d952c314b51181f4b8262ac9bf2a49898bd6",
    bucket="brainscore-storage/brainscore-vision/benchmarks/user_794",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Dense'),
)

