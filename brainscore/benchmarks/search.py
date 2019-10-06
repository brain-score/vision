import numpy as np
import ScanMatchPy
import matlab

import brainscore
from brainscore.benchmarks import BenchmarkBase
from brainscore.model_interface import BrainModel


class KlabZhang2018ObjArray(BenchmarkBase):
    def __init__(self):
        self._metric = ScanMatchPy.initialize()
        self._stimuli = brainscore.get_stimulus_set('klab.Zhang2018.search_obj_array')
        self._assemblies = []
        for i in range(1, 16):
            assembly = brainscore.get_assembly('klab.Zhang2018.search_obj_array_sub_' + str(i))
            self._assemblies.append(assembly)

        super(KlabZhang2018ObjArray, self).__init__(
            identifier='klab.Zhang2018.search_obj_array',
            parent='visual_search',
            paper_link='https://doi.org/10.1038/s41467-018-06217-x')

    def __call__(self, candidate: BrainModel):
        self._metric.initialize()
        candidate.start_task(self._stimuli)
        cumm_perf, fix_model = candidate.look_at(self._stimuli)

        scores = []
        for fix_sub in self._assemblies:
            fix1 = matlab.int32(fix_model.values.tolist())
            fix2 = matlab.int32(fix_sub.values.tolist())
            score = self._metric.findScore(fix1, fix2)
            scores.append(score)

        scores = np.asarray(scores)
        score = np.mean(scores)
        self._metric.terminate()

        return score
