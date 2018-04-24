import mkgu


def load_hvm(group=lambda hvm: hvm.multi_groupby(['obj', 'id'])):
    assy_hvm = mkgu.get_assembly(name="HvM")
    hvm_it_v6 = assy_hvm.sel(var="V6").sel(region="IT")
    hvm_it_v6.load()
    hvm_it_v6 = group(hvm_it_v6)
    hvm_it_v6 = hvm_it_v6.mean(dim="presentation").squeeze("time_bin").T
    return hvm_it_v6
