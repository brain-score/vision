import numpy as np
import ScanMatchPy
import matlab

import brainscore
import brainio_collection
from brainscore.benchmarks import Benchmark, ceil_score
from brainscore.model_interface import BrainModel
from brainscore.metrics import Score
from brainscore.utils import fullname
import logging
from tqdm import tqdm

class _KlabZhang2018ObjSearch(Benchmark):
    def __init__(self, ceil_score=None, assembly_name=None, identifier_suffix=""):
        self.human_score = Score([ceil_score, np.nan],
                        coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        self._version = 1
        self._identifier='klab.Zhang2018.ObjSearch-' + identifier_suffix
        self.parent='visual_search',
        self.paper_link='https://doi.org/10.1038/s41467-018-06217-x'
        self._assemblies = brainscore.get_assembly(assembly_name)
        self._stimuli = self._assemblies.stimulus_set
        self.fix = [[640, 512],
                     [365, 988],
                     [90, 512],
                     [365, 36],
                     [915, 36],
                     [1190, 512],
                     [915, 988]]
        self.max_fix = 6
        self.data_len = 300
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, candidate: BrainModel):
        self._metric = ScanMatchPy.initialize()

        self._logger.info("## Starting visual search task...")
        candidate.start_task(BrainModel.Task.object_search, fix=self.fix, max_fix=self.max_fix, data_len=self.data_len)
        self.cumm_perf, self.saccades = candidate.look_at(self._stimuli)
        # in saccades the first 7 index are the saccades whereas the last index denotes the index at which the target was found
        fix_model = self.saccades[:,:7,:] # first 7 saccades
        I_fix_model = self.saccades[:,7,:1]  # index at which the target was found
        fix1 = matlab.int32(fix_model.tolist())
        I_fix1 = matlab.int32(I_fix_model.tolist())
        self._logger.info("## Search task done...\n")

        self._logger.info("## Calculating score...")
        scores = []
        for sub_id in tqdm(range(15), desc="comparing with human data: "):
            data_human = self._assemblies.values[sub_id*self.data_len:(sub_id+1)*self.data_len]
            fix_human = data_human[:,:7,:]
            I_fix_human = data_human[:,7,:1]
            fix2 = matlab.int32(fix_human.tolist())
            I_fix2 = matlab.int32(I_fix_human.tolist())
            score = self._metric.findScore(fix1, fix2, I_fix1, I_fix2)
            scores.append(score)

        scores = np.asarray(scores)

        self.raw_score = np.mean(scores)
        self.std = np.std(scores)/np.sqrt(scores.shape[0])

        self.model_score = Score([self.raw_score, self.std],
                        coords={'aggregation': ['center', 'error']}, dims=['aggregation'])

        self._metric.terminate()

        ceiled_score = ceil_score(self.model_score, self.ceiling)
        self._logger.info("## Score calculated...\n")

        return ceiled_score

    @property
    def identifier(self):
        return self._identifier

    @property
    def version(self):
        return self._version

    @property
    def ceiling(self):
        return self.human_score

    def get_raw_data(self):
        return self.cumm_perf, self.saccades
    
def KlabZhang2018ObjSearchObjArr():
    return _KlabZhang2018ObjSearch(ceil_score=0.4411, assembly_name='klab.Zhang2018search_obj_array', identifier_suffix='objarr')

class _KlabZhang2018VisualSearch(Benchmark):
    def __init__(self, ceil_score=None, assembly_name=None, identifier_suffix=""):
        self._logger = logging.getLogger(fullname(self))

        self.human_score = Score([ceil_score, np.nan],
                        coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        self._version = 1
        self._identifier='klab.Zhang2018.VisualSearch-' + identifier_suffix
        self.parent = 'visual_search'
        self.paper_link='https://doi.org/10.1038/s41467-018-06217-x'
        self._assemblies = brainscore.get_assembly(assembly_name)
        self._stimuli = self._assemblies.stimulus_set

        self.max_fix = self._assemblies.fixation.values.shape[0] - 2
        self.num_sub = np.max(self._assemblies.subjects.values)
        self.data_len = int(self._assemblies.presentation.values.shape[0]/self.num_sub)
        self.ior_size = 100

    def __call__(self, candidate: BrainModel):
        self._metric = ScanMatchPy.initialize()

        self._logger.info("## Starting visual search task...")
        candidate.start_task(BrainModel.Task.visual_search, max_fix=self.max_fix, data_len=self.data_len, ior_size=self.ior_size)
        self.cumm_perf, self.saccades = candidate.look_at(self._stimuli)
        # in saccades the last index denotes the index at which the target was found
        fix_model = self.saccades[:,:self.max_fix+1,:] # first n saccades
        I_fix_model = self.saccades[:,self.max_fix+1,:1]  # index at which the target was found
        fix1 = matlab.int32(fix_model.tolist())
        I_fix1 = matlab.int32(I_fix_model.tolist())
        self._logger.info("## Search task done...\n")

        self._logger.info("## Calculating score...")
        scores = []
        for sub_id in tqdm(range(self.num_sub), desc="comparing with human data: "):
            data_human = self._assemblies.values[sub_id*self.data_len:(sub_id+1)*self.data_len]
            fix_human = data_human[:,:self.max_fix+1,:]
            I_fix_human = data_human[:,self.max_fix+1,:1]
            fix2 = matlab.int32(fix_human.tolist())
            I_fix2 = matlab.int32(I_fix_human.tolist())
            score = self._metric.findScore(fix1, fix2, I_fix1, I_fix2)
            scores.append(score)

        scores = np.asarray(scores)

        self.raw_score = np.mean(scores)
        self.std = np.std(scores)/np.sqrt(scores.shape[0])

        self.model_score = Score([self.raw_score, self.std],
                        coords={'aggregation': ['center', 'error']}, dims=['aggregation'])

        self._metric.terminate()

        ceiled_score = ceil_score(self.model_score, self.ceiling)
        self._logger.info("## Score calculated...\n")

        return ceiled_score

    @property
    def identifier(self):
        return self._identifier

    @property
    def version(self):
        return self._version

    @property
    def ceiling(self):
        return self.human_score

    def get_raw_data(self):
        return self.cumm_perf, self.saccades
    
def KlabZhang2018VisualSearchWaldo():
    return _KlabZhang2018VisualSearch(ceil_score=0.3519, assembly_name='klab.Zhang2018search_waldo', identifier_suffix="waldo")

def KlabZhang2018VisualSearchNaturaldesign():
    return _KlabZhang2018VisualSearch(ceil_score=0.3953, assembly_name='klab.Zhang2018search_naturaldesign', identifier_suffix="naturaldesign")
