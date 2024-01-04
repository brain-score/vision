import os
import yaml
from yacs.config import CfgNode as CN

base_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'physion_pretraining_size_base.yaml')
cfg = yaml.safe_load(open(base_file, 'rb'))
# only assign specific keys to avoid overwriting anything
cfg['DATA_SPACE']['KWARGS']['pretraining']['train']['suffix'] = '_size63_split2'

cfg['CONFIG']['EXPERIMENT_NAME'] =  'physion-pretraining-size3'
cfg['CONFIG']['POSTGRES']['DBNAME'] = 'physion-pretraining-size'
