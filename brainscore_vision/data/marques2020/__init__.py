from brainio.assemblies import PropertyAssembly

from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3, load_assembly_from_s3


BIBTEX = """@article{dapello2020simulating,
  title={Simulating a primary visual cortex at the front of CNNs improves robustness to image perturbations},
  author={Dapello, Joel and Marques, Tiago and Schrimpf, Martin and Geiger, Franziska and Cox, David and DiCarlo, James J},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={13073--13087},
  year={2020}
}"""

# movshon.cav assembly : uses dicarlo.Marques2020_size stimuli
data_registry['movshon.Cavanaugh2002a'] = lambda: load_assembly_from_s3(
    identifier="movshon.Cavanaugh2002a",
    version_id="2vAgf8I9mx0dAZ6E.v9_wQ1qjqXlBIlG",
    sha1="d4b24f4bbf8a14138a8e98391ab03796c2c05e7d",
    bucket="brainio-brainscore",
    cls=PropertyAssembly)

# movshon.freemanziemba2013 assembly : uses movshon.FreemanZiemba2013_properties stimuli
data_registry['movshon.FreemanZiemba2013_V1_properties'] = lambda: load_assembly_from_s3(
    identifier="movshon.FreemanZiemba2013_V1_properties",
    version_id="fmhJH37ozlGbGPW16B0NSgdm2fz17aEc",
    sha1="e665522d9a32e6cd6c3ebc61ff65fd3899c8d3e6",
    bucket="brainio-brainscore",
    cls=PropertyAssembly)

# devalois.DeValois1982a assembly : uses dicarlo.Marques2020_orientation stimuli
data_registry['devalois.DeValois1982a'] = lambda: load_assembly_from_s3(
    identifier="devalois.DeValois1982a",
    version_id="d_IU4t6LbkvxsaEEsu9VzhTm9jX3AhSI",
    sha1="8f4fd70e987c3c566b5b6c6bf15ef27637618b2c",
    bucket="brainio-brainscore",
    cls=PropertyAssembly)

# devalois.DeValois1982b assembly : uses dicarlo.Marques2020_spatial_frequency stimuli
data_registry['devalois.DeValois1982b'] = lambda: load_assembly_from_s3(
    identifier="devalois.DeValois1982b",
    version_id="jG7loVtkn3vsH0hT6wl4Qv24EsJJJSDt",
    sha1="611176eefdac09feaa4c07f081784c24067629e1",
    bucket="brainio-brainscore",
    cls=PropertyAssembly)

# shapley.Ringach2002 assembly : uses dicarlo.Marques2020_orientation stimuli
data_registry['shapley.Ringach2002'] = lambda: load_assembly_from_s3(
    identifier="shapley.Ringach2002",
    version_id="aMkWQinb7JEhJCJb0RhfGJegTwbya6Yw",
    sha1="0f3e19140bd9f930109879a38e574471d9576cf5",
    bucket="brainio-brainscore",
    cls=PropertyAssembly)

# schiller.Schiller1976c assembly : uses dicarlo.Marques2020_spatial_frequency stimuli
data_registry['schiller.Schiller1976c'] = lambda: load_assembly_from_s3(
    identifier="schiller.Schiller1976c",
    version_id="tJUZfDfz5G1E3UtEjptQeI7Esbg2aArD",
    sha1="58daa02ba2680e75dcb11b37eb6085c4afb6576e",
    bucket="brainio-brainscore",
    cls=PropertyAssembly)


# stimulus sets: dicarlo.Marques2020_blank
stimulus_set_registry['dicarlo.Marques2020_blank'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Marques2020_blank",
    bucket="brainio-brainscore",
    csv_sha1="c02d44b0d3e157b29c6b96c5c5a478a8b37dc70b",
    zip_sha1="5c4eef94691d6d63f8b1610918728b8531fc861e",
    csv_version_id="TPtJnOoJAKEnWSDrQ_1X9bvx_jUQkTvM",
    zip_version_id="4Zv7Mt0cGuKCV5i0uwGipVa91so_6U.V")

# stimulus sets: dicarlo.Marques2020_receptive_field
stimulus_set_registry['dicarlo.Marques2020_receptive_field'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Marques2020_receptive_field",
    bucket="brainio-brainscore",
    csv_sha1="e3681c69122b20ff3dd22486f546a26b3a97e057",
    zip_sha1="fc21a9302bb0fca85350049ab740af80f2307ece",
    csv_version_id="o6JVmVyn015uKDSqwH1Qfaek8I4AH0A2",
    zip_version_id="wFfNAHvgptFmzn58VdlABHeEPlnh54EO")

# stimulus sets: dicarlo.Marques2020_orientation
stimulus_set_registry['dicarlo.Marques2020_orientation'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Marques2020_orientation",
    bucket="brainio-brainscore",
    csv_sha1="b316c6d9c6ddb58f8f21e79ae2b59b03933c0068",
    zip_sha1="6a02466e221ac314da2db2969c89688dff64ab1b",
    csv_version_id="FuItfnJQzVkbXuC6oDZYiIa6Ye.ZrtHp",
    zip_version_id="HAlNBCxxrzdFy6g9ofCM1rG49KXMgMjH")

# stimulus sets: dicarlo.Marques2020_spatial_frequency
stimulus_set_registry['dicarlo.Marques2020_spatial_frequency'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Marques2020_spatial_frequency",
    bucket="brainio-brainscore",
    csv_sha1="31d0293c8aa1590f4680bb8a9446f56135b8d646",
    zip_sha1="6fdeee4366514b9a0ff9c52e52c13762a62ceaab",
    csv_version_id="xtKupFJyMLIqtI64H9bFikeeyycg0esU",
    zip_version_id="cPxm7TfKmFPFigPdzJjOf3gHcdKDkVXi")

# stimulus sets: dicarlo.Marques2020_size
stimulus_set_registry['dicarlo.Marques2020_size'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Marques2020_size",
    bucket="brainio-brainscore",
    csv_sha1="0fd0aeea8fa6ff2b30ee9a6a684d4600590d631f",
    zip_sha1="11122de2784218b7dd4cd0b557e62583ca7ee283",
    csv_version_id="VQsd1qLhSKqd5Wfz6tmnmypgs038.hoT",
    zip_version_id="RAfJJhR9.KQlwdiuZa6PiEiJTs62Jk.4")

# stimulus sets: movshon.FreemanZiemba2013_properties
stimulus_set_registry['movshon.FreemanZiemba2013_properties'] = lambda: load_stimulus_set_from_s3(
    identifier="movshon.FreemanZiemba2013_properties",
    bucket="brainio-brainscore",
    csv_sha1="6f7bcb5d0c01e81c9fbdcec9bf586cbb579a9b02",
    zip_sha1="c64a0d6b447676a6d5cff3d3625a5f294e62ff97",
    csv_version_id="cXG5bRIhKTFX.36oAHk3zvmC.0TreoH8",
    zip_version_id="PxoZZYyCCKJDbaoluOC.8nHZKWQ0.m8C")
