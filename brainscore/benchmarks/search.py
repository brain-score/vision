import numpy as np
import ScanMatchPy
import matlab

import brainscore
import brainio_collection
from brainscore.benchmarks import Benchmark
from brainscore.model_interface import BrainModel
from brainscore.metrics import Score
from tqdm import tqdm

class KlabZhang2018ObjArray(Benchmark):
    def __init__(self):
        self._version = 1
        self._identifier='klab.Zhang2018-ObjArray'
        self.parent='visual_search',
        self.paper_link='https://doi.org/10.1038/s41467-018-06217-x'
        self._assemblies = brainscore.get_assembly('klab.Zhang2018search_obj_array')
        self._stimuli = self._assemblies.stimulus_set
        self.human_score = 0.4411

    def __call__(self, candidate: BrainModel):
        self._metric = ScanMatchPy.initialize()
        candidate.start_task(BrainModel.Task.visual_search)
        self.cumm_perf, self.saccades = candidate.look_at(self._stimuli)
        fix_model = self.saccades[:,:7,:]
        I_fix_model = self.saccades[:,7,:1]
        fix1 = matlab.int32(fix_model.tolist())
        I_fix1 = matlab.int32(I_fix_model.tolist())

        scores = []
        for sub_id in tqdm(range(15)):
            data_human = self._assemblies.values[sub_id*300:(sub_id+1)*300]
            fix_human = data_human[:,:7,:]
            I_fix_human = data_human[:,7,:1]
            fix2 = matlab.int32(fix_human.tolist())
            I_fix2 = matlab.int32(I_fix_human.tolist())
            score = self._metric.findScore(fix1, fix2, I_fix1, I_fix2)
            scores.append(score)

        scores = np.asarray(scores)
        self.score = np.mean(scores)
        self._metric.terminate()

        ceil_score = self.ceiling
        return ceil_score

    @property
    def identifier(self):
        return self._identifier

    @property
    def version(self):
        return self._version

    @property
    def ceiling(self):
        return round(self.score/self.human_score, 3)

    def get_raw_data(self):
        return self.cumm_perf, self.saccades
