from brainio.assemblies import BehavioralAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = ""  # to appear in a future article

DATASETS = ['rgb', 'contours', 'phosphenes-12', 'phosphenes-16', 'phosphenes-21', 'phosphenes-27', 'phosphenes-35',
            'phosphenes-46', 'phosphenes-59', 'phosphenes-77', 'phosphenes-100', 'segments-12', 'segments-16',
            'segments-21', 'segments-27', 'segments-35', 'segments-46', 'segments-59', 'segments-77', 'segments-100',
            'phosphenes-all', 'segments-all']

SHA1s = {'rgb': 'cb3344bd72ab3b28e034874e43405a36eea66f34',
         'contours': '5cd333143b3de0c3350adbff53eac8d73b34d0c5',
         'phosphenes-12': '182ccfb4b0b9594e77385a42f77f1477b5190e4d',
         'phosphenes-16': '25898dd0aa6ed7a647c305a96062efdf49f4a959',
         'phosphenes-21': '25898dd0aa6ed7a647c305a96062efdf49f4a959',
         'phosphenes-27': 'bc1648e6be561f45ac25e550783603339166c881',
         'phosphenes-35': '695d663dfa0e21a7c27fe604a085e66cccea73fa',
         'phosphenes-46': 'b07b650557c053a060536784b9091a793f4099b0',
         'phosphenes-59': '86c64f85dbb74fa3f603bb67382f6e18cd5f3c58',
         'phosphenes-77': '28fac3e775e17a6638ef359834a3acec802714c3',
         'phosphenes-100': 'ea4dd583e1352abe4add5b188bf0a83678a2f1f1',
         'segments-12': '251d82a50007d978e72142cac5f922439f0bcc61',
         'segments-16': 'b57e9059b7d839cc417b860d68f731400beac4bc',
         'segments-21': 'c98774fcd19ecc7ed08e208ca9bc5b6ec3638188',
         'segments-27': '33480463a389defb65a18d73507855acfbfb69e1',
         'segments-35': 'a8bc7616b7c2e827d34bf130b32a8cc400cd7f3c',
         'segments-46': '608c7fa08b34bd578ec37cd5e8f9b6624f8bbd6a',
         'segments-59': '7833e3c01666d94b44e708f9272c64a3bae10983',
         'segments-77': 'b6ac947b26c89b6dad55618f85f07898facf22c6',
         'segments-100': '16e458a9207e44bb6c1a0e9b09b11826a8a78644',
         'phosphenes-all': '6f7e4fa38a25adb88576fe60cffbbda707660494',
         'segments-all': 'e00ab3b397afefb5377a38618c0d0421dfe80645'}


for dataset in DATASETS:
    # assembly
    data_registry[f'Scialom2024_{dataset}'] = lambda: load_assembly_from_s3(
        identifier=f'Scialom2024_{dataset}',
        version_id="",
        sha1=SHA1s[dataset],
        bucket="brainio-brainscore",
        cls=BehavioralAssembly,
        stimulus_set_loader=lambda: load_stimulus_set(f'Scialom2024_{dataset}'),
    )

    # stimulus set
    stimulus_set_registry[f'Scialom2024_{dataset}'] = lambda: load_stimulus_set_from_s3(
        identifier=f'Scialom2024_{dataset}',
        bucket="brainio-brainscore",
        csv_sha1="",
        zip_sha1="",
        csv_version_id="",
        zip_version_id="")