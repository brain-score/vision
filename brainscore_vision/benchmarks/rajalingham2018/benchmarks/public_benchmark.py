from brainscore_vision import load_metric
from .benchmark import load_assembly as load_rajalingham2018, _DicarloRajalingham2018


class RajalinghamMatchtosamplePublicBenchmark(_DicarloRajalingham2018):
    def __init__(self):
        super(RajalinghamMatchtosamplePublicBenchmark, self).__init__(
            metric=load_metric('i2n'), metric_identifier='i2n')
        self._assembly = load_rajalingham2018(access='public')
        self._ceiling_func = lambda: self._metric.ceiling(self._assembly, skipna=True)
