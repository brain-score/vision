
import os
import yaml
from yacs.config import CfgNode as CN

base_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'physion_pretraining_size_base.yaml')
cfg = yaml.safe_load(open(base_file, 'rb'))
# only assign specific keys to avoid overwriting anything
cfg['PRETRAINING']['TRAIN_STEPS'] = 10000 # increase steps for full dataset

cfg['CONFIG']['EXPERIMENT_NAME'] =  'physion-pretraining-steps'
cfg['CONFIG']['POSTGRES']['DBNAME'] = 'physion-pretraining-size'
