from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3


BIBTEX = """@article{rajalingham2018large,
  title={Large-scale, high-resolution comparison of the core visual object recognition behavior of humans, monkeys, and state-of-the-art deep artificial neural networks},
  author={Rajalingham, Rishi and Issa, Elias B and Bashivan, Pouya and Kar, Kohitij and Schmidt, Kailyn and DiCarlo, James J},
  journal={Journal of Neuroscience},
  volume={38},
  number={33},
  pages={7255--7269},
  year={2018},
  publisher={Soc Neuroscience}
}"""


# public assembly: uses objectome.public stimuli
data_registry['Rajalingham2018.public'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rajalingham2018.public",
    version_id="WEBNb7Azz4CWpzO25JanNjdPSLArltS2",
    sha1="34c6a8b6f7c523589c1861e4123232e5f7c7df4c",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('objectome.public'),
)

# private assembly: uses objectome.private stimuli
data_registry['Rajalingham2018.private'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Rajalingham2018.private",
    version_id="gwBpHTT2al4FN35Yje7MU2d_ByA_HphX",
    sha1="516f13793d1c5b72bb445bb4008448ce97a02d23",
    bucket="brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('objectome.private'),
)


# stimulus set: objectome.public - rajalingham2018
stimulus_set_registry['objectome.public'] = lambda: load_stimulus_set_from_s3(
    identifier="objectome.public",
    bucket="brainio-brainscore",
    csv_sha1="47884e17106a3be471d6481279cab33889b80850",
    zip_sha1="064f2955f98e63867755fee2e3ead8cddf6bfab8",
    csv_version_id="CwFzXLOclwodJZpInD6hc15yeTOywg6J",
    zip_version_id="g.vVSa77K84jjeto5KAryLESzxJz0yUB")

# stimulus set: objectome.private - same
stimulus_set_registry['objectome.private'] = lambda: load_stimulus_set_from_s3(
    identifier="objectome.private",
    bucket="brainio-brainscore",
    csv_sha1="ac38e8f7c08fa8294ed25a3bf84a6adb108bf3fc",
    zip_sha1="ccd39f7f9b8b4a92da06e3960d06225e46208593",
    csv_version_id=".c_qSre8xH31QWCHU_0bExY4YgWVyxXY",
    zip_version_id="TKj6DP8Aesj8IH8WDTtPVtdMCKUtAvJK")
