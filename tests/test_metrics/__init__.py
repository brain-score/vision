import brainscore


def load_hvm(group=lambda hvm: hvm.multi_groupby(['object_name', 'image_id'])):
    assembly = brainscore.get_assembly(name="dicarlo.MajajHong2015").sel(variation=6)
    assembly.load()
    assembly = group(assembly)
    assembly = assembly.mean(dim="presentation").squeeze("time_bin").T
    return assembly
