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
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="8e3eee196d5b1bca058c8dad54234fa5340eeaca",
    zip_sha1="75b5720f107eadaa9689e80bf812a173f1f099e9",
    csv_version_id="c8E0Uyj7rzX3Nq2b_kqfUchUmYGYQqXa",
    zip_version_id="R4LflCnqCdpwRSmGjqmXkOSPhS5asmQA")

# assembly
data_registry['Kato2026.Ori'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Ori',
    version_id="IcDilVH2ryDcsNlCsOkktmEwEBI2g8Oa",
    sha1="d5a22c25835da413b629d87ec77d6af834a43732",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Ori'),
)
                
# stimulus set
stimulus_set_registry['Kato2026.Fil'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Fil',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="469b6af5259450cda4eec7586ccbbcc5174aa0c4",
    zip_sha1="9b7a45311b1caa9b2f16bc97bc4ee24eb197265b",
    csv_version_id="vx5ce2ks1s0aKgkmDVISRkNX8Nb1eWds",
    zip_version_id="5MpFnt9mWZwrkjtrdhgoLuiKQJI8.xK0")

# assembly
data_registry['Kato2026.Fil'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Fil',
    version_id="N0nBBdEenaeyfZ2D7JyUMUpX868xTKVR",
    sha1="954498ac9cd106fb17443bc77ac2c00d1fe03f00",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Fil'),
)

# stimulus set
stimulus_set_registry['Kato2026.Lin'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Lin',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="36b8045876aa47151a65f16ddee645cf9473f4fb",
    zip_sha1="2b786b9901a05d1fd980ae34153536cf906a8aa1",
    csv_version_id="YL0aw.aZX1pK2UllusVUPUhs_60QbnTl",
    zip_version_id="B9OkKyr5hLMsfikSGDEYR2sSN9U1SfKe")

# assembly
data_registry['Kato2026.Lin'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Lin',
    version_id="br4S8gjoU3OlxgjNL.6n5._sDH6xgBYO",
    sha1="ce681e8afca2d39166ae2f19ba665638f794b9d1",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Lin'),
)

# stimulus set
stimulus_set_registry['Kato2026.SilTex'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.SilTex',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="ff982583d8d815247cc1a009255857772e07cd94",
    zip_sha1="6cdc21439438e126f3ab401bb05af84300a4fe1e",
    csv_version_id="H9jSJKHiCvXur81Ri5c0oLvOgvUbYA0L",
    zip_version_id="7v.w1CApejxHd8PqNrt7apuSw9JvwU5Z")

# assembly
data_registry['Kato2026.SilTex'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.SilTex',
    version_id="PyJkPnNPsSDVPS.Ov25h72lA5HON1bb4",
    sha1="ce1474dea31dd0b7469ec634e5e61da55e414502",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.SilTex'),
)

# stimulus set
stimulus_set_registry['Kato2026.Sil'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Sil',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="13a42a068aa79fd533fd7b5f00a09f67053b4425",
    zip_sha1="0d02bea9d282d020d9b20ceb7daa50d185cdefe3",
    csv_version_id="YhGqxxegXosVc3VArhgMPtLvX1qzbBJA",
    zip_version_id="u.yDFDHxFkT6YG3KcuA7V5SrKKrdAZEX")

# assembly
data_registry['Kato2026.Sil'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Sil',
    version_id="xbEvAod60hmj6.59PqJkxT3E4gLMCOMr",
    sha1="4a21ce43ca5f7dc0b8d9c3d048ea2dda2ca7f47a",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Sil'),
)

# stimulus set
stimulus_set_registry['Kato2026.Tex'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Tex',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="cd533001022f56bef5b39ce43b00a4a4293a81d1",
    zip_sha1="fb6564cae35ca18172025b46cd653f4721015b3d",
    csv_version_id="ndXeBEGY2iZqcFb32TftiL5sd1t5mpp4",
    zip_version_id="UK7ewdy71NKV7qL9jLMf2mjvhxjbvpiS")

# assembly
data_registry['Kato2026.Tex'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Tex',
    version_id="96guBSPBcQnvhWBolcHRd85cW2SuAqr0",
    sha1="cb4d428412d4334f57e9b7b14c7e7d888bf838db",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Tex'),
)

# stimulus set
stimulus_set_registry['Kato2026.LinFil'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.LinFil',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="9fbeb46e9a2a5ac0573ebf4ce6bb20fb9ddf3a33",
    zip_sha1="2ac934e65c3645b960c3e1126921cea783da6479",
    csv_version_id="YHi84d2HphKadlfILlfpwLnZCeVscOcZ",
    zip_version_id="N2tXSQgWGdyaZ224QwGniwqruq1CLyrW")

# assembly
data_registry['Kato2026.LinFil'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.LinFil',
    version_id="i46dCvs_.gUhWrUV4MxesQNIOgxNVrQQ",
    sha1="87f0227c3507e8cf1172ddf5398036f098a40c96",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.LinFil'),
)

# stimulus set
stimulus_set_registry['Kato2026.Sparse'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Sparse',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="990d317b8d526590257d4acc0abf70e89a4f6661",
    zip_sha1="67569fca4b89b9ae4e893f4eeb71e3b2c2c67808",
    csv_version_id="Qklfu769cHwUFh4Di18LDieEmE8F2fYW",
    zip_version_id="hNOh64MuEOc_awIbxvNZPVDExCsWkyhY")

# assembly
data_registry['Kato2026.Sparse'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Sparse',
    version_id="DaFbWgSf3zK1SL2zwr4X4lt7g9zXE4gk",
    sha1="e933f23d2c09a2107b292657063a99d0a931119e",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Sparse'),
)

# stimulus set
stimulus_set_registry['Kato2026.Dense'] = lambda: load_stimulus_set_from_s3(
    identifier='Kato2026.Dense',
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="4e1b9387371d416a0c5758cd78ee683c23c7e191",
    zip_sha1="7557f8e1542cd14edd1f4cf79264ea4851861651",
    csv_version_id="uGiSwGUkn8GY9eCTxowtp3wd_.LudjPk",
    zip_version_id="hTYMs8YJThDsc7La1zb1oHOblyURWNjX")

# assembly
data_registry['Kato2026.Dense'] = lambda: load_assembly_from_s3(
    identifier='Kato2026.Dense',
    version_id="L6p2mXkQwISAbN5_J7kmoEYnmVpnfFoi",
    sha1="49b2d952c314b51181f4b8262ac9bf2a49898bd6",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kato2026.Dense'),
)

