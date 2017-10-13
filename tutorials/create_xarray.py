import xarray as xr
from streams.envs import hvm
import mkgu
import numpy as np
import mkgu.metrics
from mkgu.assemblies import ModelFeaturesAssembly


assy_hvm = mkgu.get_assembly(name="HvM")
neural = assy_hvm.sel().sel(region="IT")
import ipdb; ipdb.set_trace()
neural = neural.groupby('image_id').mean(dim='presentation').squeeze('time_bin').T

ims = assy_hvm.stimulus_set


hvmit = hvm.HvM()
model_feats = hvmit.model(name='basenet6', layer='aIT', pca=True)
coords = {'image_id': ('presentation', hvmit.meta.id), 'obj': ('presentation', hvmit.meta.obj)}
        #   'nid': ('neuroid', np.arange(1000).astype(int))}
mf = ModelFeaturesAssembly(model_feats, dims=['presentation', 'neuroid'], coords=coords)

mkgu.metrics.neural_fits(neural, mf, 'obj')

