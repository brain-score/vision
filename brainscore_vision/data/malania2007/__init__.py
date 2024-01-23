from brainio.assemblies import PropertyAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3


BIBTEX = """@article{malania2007,
            author = {Malania, Maka and Herzog, Michael H. and Westheimer, Gerald},
            title = "{Grouping of contextual elements that affect vernier thresholds}",
            journal = {Journal of Vision},
            volume = {7},
            number = {2},
            pages = {1-1},
            year = {2007},
            issn = {1534-7362},
            doi = {10.1167/7.2.1},
            url = {https://doi.org/10.1167/7.2.1}
        }"""

# 'equal-2'
# assembly
data_registry['Malania2007_equal-2'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_equal-2',
    version_id="",
    sha1="",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Malania2007_equal-2'),
)

# stimulus set
stimulus_set_registry['Malania2007_equal-2'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-2',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# stimulus set fitting stimuli
stimulus_set_registry['Malania2007_equal-2_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-2_fit',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# 'equal-16'
# assembly
data_registry['Malania2007_equal-16'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_equal-16',
    version_id="",
    sha1="",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Malania2007_equal-16'),
)

# stimulus set
stimulus_set_registry['Malania2007_equal-16'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-16',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# stimulus set fitting stimuli
stimulus_set_registry['Malania2007_equal-16_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_equal-16_fit',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# 'long-2'
# assembly
data_registry['Malania2007_long-2'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_long-2',
    version_id="",
    sha1="",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Malania2007_long-2'),
)

# stimulus set
stimulus_set_registry['Malania2007_long-2'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-2',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# stimulus set fitting stimuli
stimulus_set_registry['Malania2007_long-2_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-2_fit',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# 'long-16'
# assembly
data_registry['Malania2007_long-16'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_long-16',
    version_id="",
    sha1="",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Malania2007_long-16'),
)

# stimulus set
stimulus_set_registry['Malania2007_long-16'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-16',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# stimulus set fitting stimuli
stimulus_set_registry['Malania2007_long-16_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_long-16_fit',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# 'short-2'
# assembly
data_registry['Malania2007_short-2'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-2',
    version_id="",
    sha1="",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Malania2007_short-2'),
)

# stimulus set
stimulus_set_registry['Malania2007_short-2'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-2',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# stimulus set fitting stimuli
stimulus_set_registry['Malania2007_short-2_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-2_fit',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# 'short-4'
# assembly
data_registry['Malania2007_short-4'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-4',
    version_id="",
    sha1="",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Malania2007_short-4'),
)

# stimulus set
stimulus_set_registry['Malania2007_short-4'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-4',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# stimulus set fitting stimuli
stimulus_set_registry['Malania2007_short-4_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-4_fit',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# 'short-6'
# assembly
data_registry['Malania2007_short-6'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-6',
    version_id="",
    sha1="",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Malania2007_short-6'),
)

# stimulus set
stimulus_set_registry['Malania2007_short-6'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-6',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# stimulus set fitting stimuli
stimulus_set_registry['Malania2007_short-6_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-6_fit',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# 'short-8'
# assembly
data_registry['Malania2007_short-8'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-8',
    version_id="",
    sha1="",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Malania2007_short-8'),
)

# stimulus set
stimulus_set_registry['Malania2007_short-8'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-8',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# stimulus set fitting stimuli
stimulus_set_registry['Malania2007_short-8_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-8_fit',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# 'short-16'
# assembly
data_registry['Malania2007_short-16'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_short-16',
    version_id="",
    sha1="",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Malania2007_short-16'),
)

# stimulus set
stimulus_set_registry['Malania2007_short-16'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-16',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# stimulus set fitting stimuli
stimulus_set_registry['Malania2007_short-16_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_short-16_fit',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# 'vernier-only'
# assembly
data_registry['Malania2007_vernier-only'] = lambda: load_assembly_from_s3(
    identifier='Malania2007_vernier-only',
    version_id="",
    sha1="",
    bucket="brainio-brainscore",
    cls=PropertyAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Malania2007_vernier-only'),
)

# stimulus set
stimulus_set_registry['Malania2007_vernier-only'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_vernier-only',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")

# stimulus set fitting stimuli
stimulus_set_registry['Malania2007_vernier-only_fit'] = lambda: load_stimulus_set_from_s3(
    identifier='Malania2007_vernier-only_fit',
    bucket="brainio-brainscore",
    csv_sha1="",
    zip_sha1="",
    csv_version_id="",
    zip_version_id="")