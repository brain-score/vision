from brainio.assemblies import DataAssembly
from brainscore_vision import data_registry, stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3
from brainscore_vision.data_helpers.lookup_legacy import version_id_df, build_filename

BIBTEX = """@article{CICHY201912,
title = {The spatiotemporal neural dynamics underlying perceived similarity for real-world objects},
journal = {NeuroImage},
volume = {194},
pages = {12-24},
year = {2019},
issn = {1053-8119},
doi = {https://doi.org/10.1016/j.neuroimage.2019.03.031},
url = {https://www.sciencedirect.com/science/article/pii/S1053811919302083},
author = {Radoslaw M. Cichy and Nikolaus Kriegeskorte and Kamila M. Jozwik and Jasper J.F. {van den Bosch} and Ian Charest},
keywords = {fMRI, MEG, Object recognition, Perceived similarity, Visual perception},
abstract = {The degree to which we perceive real-world objects as similar or dissimilar structures our perception and guides categorization behavior. Here, we investigated the neural representations enabling perceived similarity using behavioral judgments, fMRI and MEG. As different object dimensions co-occur and partly correlate, to understand the relationship between perceived similarity and brain activity it is necessary to assess the unique role of multiple object dimensions. We thus behaviorally assessed perceived object similarity in relation to shape, function, color and background. We then used representational similarity analyses to relate these behavioral judgments to brain activity. We observed a link between each object dimension and representations in visual cortex. These representations emerged rapidly within 200 ms of stimulus onset. Assessing the unique role of each object dimension revealed partly overlapping and distributed representations: while color-related representations distinctly preceded shape-related representations both in the processing hierarchy of the ventral visual pathway and in time, several dimensions were linked to high-level ventral visual cortex. Further analysis singled out the shape dimension as neither fully accounted for by supra-category membership, nor a deep neural network trained on object categorization. Together our results comprehensively characterize the relationship between perceived similarity of key object dimensions and neural activity.}
}"""

# extract version ids from version_ids csv
assembly_version = version_id_df.at[build_filename('aru.Cichy2019', '.nc'), 'version_id']
csv_version = version_id_df.at[build_filename('aru.Cichy2019', '.csv'), 'version_id']
zip_version = version_id_df.at[build_filename('aru.Cichy2019', '.zip'), 'version_id']

# assembly
data_registry['aru.Cichy2019'] = lambda: load_assembly_from_s3(
    identifier="aru.Cichy2019",
    version_id="83yFgsx5rc6pgpra4UiMxDDvgaV6ytU8",
    sha1="701e63be62b642082d476244d0d91d510b3ff05d",
    bucket="brainio-brainscore",
    cls=DataAssembly)

# stimulus set
stimulus_set_registry['aru.Cichy2019'] = lambda: load_stimulus_set_from_s3(
    identifier="aru.Cichy2019",
    bucket="brainio-brainscore",
    csv_sha1="281c4d9d0dd91a2916674638098fe94afb87d29a",
    zip_sha1="d2166dd9c2720cb24bc520f5041e6830779c0240",
    csv_version_id="ZtjaClAgoeJPA24WVCsd2IywsML5_pBc",
    zip_version_id="UmONlzEKwa7MaLWIKpWNONe9QSXM8NI6")
