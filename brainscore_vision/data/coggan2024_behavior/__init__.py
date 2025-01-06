# Created by David Coggan on 2024 06 23

from brainio.assemblies import BehavioralAssembly
from brainscore_vision import (
    stimulus_set_registry, data_registry, load_stimulus_set)
from brainscore_vision.data_helpers.s3 import (
    load_assembly_from_s3, load_stimulus_set_from_s3)

# stimulus set
stimulus_set_registry['Coggan2024_behavior'] = lambda: load_stimulus_set_from_s3(
    identifier="tong.Coggan2024_behavior",
    bucket="brainscore-storage/brainio-brainscore",
    csv_sha1="01c312c4c16f7acc5afddaafcf826e5af58b13e2",
    zip_sha1="1c070b88fa45e9e69d58f95466cb6406a45a4873",
    csv_version_id="null",
    zip_version_id="null")

# fitting stimuli
stimulus_set_registry['Coggan2024_behavior_fitting'] = lambda: (
    load_stimulus_set_from_s3(
        identifier="tong.Coggan2024_behavior_fitting",
        bucket="brainscore-storage/brainio-brainscore",
        csv_sha1="136e48992305ea78a4fb77e9dfc75dcf01e885d0",
        zip_sha1="24e68f5ba2f8f2105daf706307642637118e7d36",
        csv_version_id="null",
        zip_version_id="null"))

# behavioral data
data_registry['Coggan2024_behavior'] = lambda: load_assembly_from_s3(
    identifier="tong.Coggan2024_behavior",
    version_id="null",
    sha1="c1ac4a268476c35bbe40081358667a03d3544631",
    bucket="brainscore-storage/brainio-brainscore",
    cls=BehavioralAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Coggan2024_behavior'),
)
