from brainscore import score_model
from candidate_models.model_commitments import brain_translated_pool
import os
from brainscore.benchmarks.domain_transfer_analysis import _OOD_BehavioralBenchmark


custom_cache_directory = "../work/upschrimpf1/bocini"
os.environ['RESULTCACHING_HOME'] = custom_cache_directory
os.environ['RESULTCACHING_DISABLE'] = '0'

identifier = 'alexnet' #'resnet50' 'voneresnet-50-robust'


model = brain_translated_pool[identifier]

benchmark = _OOD_BehavioralBenchmark()
score = benchmark(model)

import pdb; pdb.set_trace()
print(score)

