import numpy as np
import random
import pandas as pd
import brainscore
from brainscore.benchmarks import BenchmarkBase, Score
from brainscore.metrics.dimensionality import Dimensionality
from brainscore.model_interface import BrainModel
from tqdm import tqdm 
from brainio.stimuli import StimulusSet
from model_tools.brain_transformation import ModelCommitment

BIBTEX = """@inproceedings{
            islam2021shape,
            title={Shape or Texture: Understanding Discriminative Features in {\{}CNN{\}}s},
            author={Md Amirul Islam and Matthew Kowal and Patrick Esser and Sen Jia and Bj{\"o}rn Ommer and Konstantinos G. Derpanis and Neil Bruce},
            booktitle={International Conference on Learning Representations},
            year={2021},
            url={https://openreview.net/forum?id=NcFEZOi-rLa}
        }""" 


TIME_BIN_ST, TIME_BIN_END = 70, 170  # standard core object recognition response, following Majaj*, Hong*, et al. 2015
SEED = 1751 #turn this benchmark into a deterministic one 

class Islam2021Dimensionality(BenchmarkBase):
    def __init__(self,region,factor,deterministic=True):
        assert factor in ["shape","texture","residual"] 
        factor_idx = {"shape":"shape", "texture":"texture", "residual":"residual"}[factor]
        assert region in ["V1","V2","V4","IT"]
        self.stimulus_set  = brainscore.get_stimulus_set("Islam2021")
        self.region = region
        self.deterministic = deterministic
        self._metric = Dimensionality(factor_idx)
        self._number_of_trials = 1        
        super(Islam2021Dimensionality, self).__init__( #! not too sure about this
            identifier=f'Islam2021-{region + "_" + factor + "_dimensionality"}', version=1,
            ceiling_func=lambda: Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation']),
            parent='Islam2021',
            bibtex=BIBTEX)
        
    def set_generator(self):
        if self.deterministic:
            self.generator = random.Random(SEED)
        else:
            self.generator = random.Random()
    
    def __call__(self, candidate: BrainModel):
        self.set_generator()
        candidate.start_recording(self.region,[(TIME_BIN_ST, TIME_BIN_END)]) 
        assembly = candidate.look_at(self.stimulus_set)
        factors, assembly1, assembly2 = self.get_assembly_sets(assembly) 
        assembly1 = self.prepare_assembly(assembly1,factors)
        assembly2 = self.prepare_assembly(assembly2,factors)
        score = self._metric(assembly1, assembly2)          
        return score

    def get_assembly_sets(self,assembly,samples=None):
        textures = list(set(self.stimulus_set["texture"].values))
        textures.sort()
        assert len(textures) == 5
        shapes = list(set(self.stimulus_set["shape"].values))
        shapes.sort()
        assert len(shapes) == 20
        factors, indexes1, indexes2 = [], [], []
        if samples is None:
            samples = len(self.stimulus_set) 
        for idx1 in tqdm(range(samples)):
            factor, idx2 = self.get_index_pair(idx1,textures,shapes)
            indexes1.append(idx1)
            indexes2.append(idx2)
            factors.append(factor)
        return np.array(factors), assembly[:,np.array(indexes1)], assembly[:,np.array(indexes2)]
                        
    def get_index_pair(self,idx1,textures,shapes):
        sample1 = self.stimulus_set.iloc[idx1]
        factor = self.generator.choice(["shape", "texture"])
        if factor == "shape": # same shape, different texture
            list_possible_textures = textures.copy()
            list_possible_textures.remove(sample1["texture"])
            texture2 = self.generator.choice(list_possible_textures)
            possible_samples_cond = \
                (self.stimulus_set["original_image_id"] == sample1["original_image_id"]) & \
                (self.stimulus_set["texture"] == texture2)
            idx2 = np.where(possible_samples_cond)[0].item()
            sample2 = self.stimulus_set.iloc[idx2]
            assert sample2["shape"] == sample1["shape"] and sample1["texture"] != sample2["texture"]
            
        else: #different shape, same texture
            list_possible_shapes = shapes.copy()
            list_possible_shapes.remove(sample1["shape"])
            shape2 = self.generator.choice(list_possible_shapes)
            possible_samples_cond = \
                (self.stimulus_set["texture"] == sample1["texture"]) & \
                (self.stimulus_set["shape"] == shape2)
            possible_indexes = np.where(possible_samples_cond)[0]
            idx2 = self.generator.choice(possible_indexes)
            sample2 = self.stimulus_set.iloc[idx2]
            assert sample2["shape"] != sample1["shape"] and sample1["texture"] == sample2["texture"]
        return factor, idx2
  
    def prepare_assembly(self,assembly,factors):
        # prepare data assembly for the dimensionality metric
        assert assembly.shape[1] == len(factors)
        dim = assembly.dims[1]
        assembly = assembly.assign_coords(factor = (dim,factors))
        assembly = assembly.T
        return assembly    
       
 
def Islam2021Dimensionality_V1_Shape():
    return Islam2021Dimensionality("V1","shape")
 
def Islam2021Dimensionality_V1_Texture():
    return Islam2021Dimensionality("V1","texture")
    
def Islam2021Dimensionality_V2_Shape():
    return Islam2021Dimensionality("V2","shape")
    
def Islam2021Dimensionality_V2_Texture():
    return Islam2021Dimensionality("V2","texture")
 
def Islam2021Dimensionality_V4_Shape():
    return Islam2021Dimensionality("V4","shape")
    
def Islam2021Dimensionality_V4_Texture():
    return Islam2021Dimensionality("V4","texture")
    
def Islam2021Dimensionality_IT_Shape():
    return Islam2021Dimensionality("IT","shape")
    
def Islam2021Dimensionality_IT_Texture():
    return Islam2021Dimensionality("IT","texture")