from brainscore import score_model
from candidate_models.model_commitments import brain_translated_pool
import os
from brainscore.benchmarks.domain_transfer_neural import IT_pls


custom_cache_directory = "../work/upschrimpf1/bocini"
os.environ['RESULTCACHING_HOME'] = custom_cache_directory
os.environ['RESULTCACHING_DISABLE'] = '0'

#identifier = 'alexnet' #'resnet50' 'voneresnet-50-robust' 'dcgan' 'resnet-18', prednet
custom_cache_directory = "../work/upschrimpf1/bocini"
os.environ['RESULTCACHING_HOME'] = custom_cache_directory
identifier = 'alexnet' 


model = brain_translated_pool[identifier]
benchmark = IT_pls()
score = benchmark(model)

# score = score_model(model_identifier=identifier, model=model, benchmark_identifier='bocini-nsd-2023.whole_brain-pls')
import pdb; pdb.set_trace()
print(score)

