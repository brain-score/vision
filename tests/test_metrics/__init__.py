import brainio


def load_hvm(group=lambda hvm: hvm.multi_groupby(['object_name', 'stimulus_id'])):
    assembly = brainio.get_assembly(identifier="dicarlo.MajajHong2015").sel(variation=6)
    assembly.load()
    assembly = group(assembly)
    assembly = assembly.mean(dim="presentation").squeeze("time_bin").T
    return assembly
