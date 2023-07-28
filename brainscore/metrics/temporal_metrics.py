from assembly_utils import apply_over_dims, align_dims, get_coords_except_dims
from brainscore.metrics.regression import CrossRegressedCorrelation
from brainio.assemblies import DataAssembly
import numpy as np


class TemporalMappingWrapper:
    def __init__(self, callable, time_dim="time_bin"):
        self.time_dim = time_dim
        self.callable = callable

    def __call__(self, *asms):
        asms = align_dims(*asms, dims=[self.time_dim], dims_must_equal=False)
        return apply_over_dims(self.callable, *asms, dims=[self.time_dim])


class StaticMappingWrapper:
    def __init__(self, callable, time_dim="time_bin", presentation_dim="presentation", same_nsample_as_temporal=False):
        # if same_nsample_as_temporal, the number of samples is forced to be the same as that of presentation_dim
        self.time_dim = time_dim
        self.pres_dim = presentation_dim
        self.aligned_dims = [self.pres_dim, self.time_dim]
        self.callable = callable
        self.resample = same_nsample_as_temporal

    def __call__(self, *asms):
        aligned_dims = self.aligned_dims
        asms = align_dims(*asms, dims=aligned_dims, dims_must_equal=False)
        # choose the fist because we already aligned
        pres_size = ps = asms[0].sizes[self.pres_dim]
        time_size = ts = asms[0].sizes[self.time_dim]
        pres_coord = asms[0].coords[self.pres_dim]
        time_coord = asms[0].coords[self.time_dim]
        other_coords = [get_coords_except_dims(
            asm, dims=aligned_dims) for asm in asms]
        other_dims = [[d for d in asm.dims if d not in aligned_dims]
                      for asm in asms]
        fake_pres_coord = self._make_fake_pres_coord(pres_coord, ts)
        fake_asms = [DataAssembly(asm.values.reshape(ps*ts, *asm.values.shape[2:]), dims=[self.pres_dim, *od], coords={**fake_pres_coord, **oc})
                     for asm, oc, od in zip(asms, other_coords, other_dims)]
        if self.resample:
            indices = np.random.randint(0, ps*ts, ps)
            fake_asms = [asm[indices] for asm in fake_asms]
        fake_outs = self.callable(*fake_asms)
        if fake_outs is None:
            return
        if not isinstance(fake_outs, list) or not isinstance(fake_outs, tuple):
            fake_outs = [fake_outs]
        outs = [DataAssembly(out.values.reshape(ps, ts, *out.values.shape[1:]),
                coords={self.pres_dim: pres_coord, self.time_dim: time_coord, **get_coords_except_dims(out, dims=aligned_dims)})
                for out in fake_outs]
        if len(outs) == 1:
            return outs[0]
        else:
            return outs

    def _make_fake_pres_coord(self, pres_coord, time_size):
        levels = pres_coord.variable.level_names
        levels = levels or [self.pres_dim]  # if levels are None
        fake_coord = {}
        for level in levels:
            vals = pres_coord.coords[level].values
            new_vals = []
            for v in vals:
                for t in range(time_size):
                    new_vals.append(f"{v}_{t}")

            fake_coord[level] = (self.pres_dim, new_vals)

        # Multiindex bug: single level
        if len(levels) == 1:
            fake_coord[self.pres_dim] = (self.pres_dim, new_vals)

        return fake_coord


def wrap_callable(callable, wrapper, wrapper_kwargs):
    def wrapped(*args, **kwargs):
        return wrapper(callable, **wrapper_kwargs)(*args, **kwargs)
    return wrapped


def TemporalMapping(mapping, **wrapper_kwargs):
    return wrap_callable(mapping, TemporalMappingWrapper, wrapper_kwargs)


def StaticMapping(mapping, **wrapper_kwargs):
    return wrap_callable(mapping, StaticMappingWrapper, wrapper_kwargs)


def TemporalCrossRegressedCorrelation(*args, time_dim="time_bin", **kwargs):
    crc = CrossRegressedCorrelation(*args, **kwargs)
    return TemporalMapping(crc, time_dim=time_dim)


def _StaticCrossRegressedCorrelation(*args, time_dim="time_bin", presentation_dim="presentation", same_nsample_as_temporal=True, **kwargs):
    crc = CrossRegressedCorrelation(*args, **kwargs)
    reg = getattr(crc, "regression")
    setattr(reg, "fit", StaticMapping(getattr(reg, "fit"), time_dim=time_dim,
            presentation_dim=presentation_dim, same_nsample_as_temporal=same_nsample_as_temporal))
    setattr(reg, "predict", StaticMapping(getattr(reg, "predict"), time_dim=time_dim,
            presentation_dim=presentation_dim, same_nsample_as_temporal=False))  # TemporalMapping also works, but less efficient
    # we still compute correlation per time stamp
    setattr(crc, "correlation", TemporalMapping(getattr(crc, "correlation")))
    return crc


def StaticCrossRegressedCorrelationSameSample(*args, **kwargs):
    return _StaticCrossRegressedCorrelation(same_nsample_as_temporal=True, *args, **kwargs)


def StaticCrossRegressedCorrelationAllSample(*args, **kwargs):
    return _StaticCrossRegressedCorrelation(same_nsample_as_temporal=False, *args, **kwargs)


if __name__ == "__main__":
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr
    from brainscore.metrics.regression import linear_regression, pearsonr_correlation
    from brainscore.metrics.xarray_utils import XarrayRegression

    X = np.random.randn(4, 5, 3)
    X_asm = DataAssembly(X, dims=["presentation", "time_bin", "neuroid"],
                         coords={"stimulus_id": ("presentation", [3,2,1,0]), "std": ("presentation", range(4)), "neuroid_id": ("neuroid", range(3)), "ntd": ("neuroid", range(3))})

    Y = np.random.randn(4, 5, 2)
    Y_asm = DataAssembly(Y, dims=["presentation", "time_bin", "neuroid"],
                         coords={"stimulus_id": ("presentation", range(4)), "std": ("presentation", range(4)), "neuroid_id": ("neuroid", range(2)), "ntd": ("neuroid", range(2))})

    # Static case

    lr_sk = LinearRegression()
    lr_sk.fit(X[::-1].reshape(20, 3), Y.reshape(20, 2))  # stimulus_id [3,2,1,0] -> [0,1,2,3] to align with Y
    Y_pred_sk = lr_sk.predict(X.reshape(20, 3)).reshape(4, 5, 2)

    def StaticEstimatorWrapper(regression):
        setattr(regression, "fit", StaticMapping(
            getattr(regression, "fit"), same_nsample_as_temporal=False))
        setattr(regression, "predict", TemporalMapping(
            getattr(regression, "predict")))
        return regression

    lr_asm = LinearRegression()
    lr_asm = StaticEstimatorWrapper(XarrayRegression(lr_asm))  # default stimulus_coords is stimulus_id
    lr_asm.fit(X_asm, Y_asm)
    Y_pred_asm = lr_asm.predict(X_asm).transpose(
        "presentation", "time_bin", "neuroid").values

    assert (Y_pred_sk == Y_pred_asm).all()
