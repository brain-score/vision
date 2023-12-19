from brainscore import score_model
from candidate_models.model_commitments import brain_translated_pool
import os
from brainscore.benchmarks.domain_transfer_analysis import OOD_BehavioralBenchmark
from brainscore.benchmarks.public_benchmarks import MajajHongITPublicBenchmark
from model_tools.brain_transformation import ModelCommitment
import importlib


custom_cache_directory = "../work/upschrimpf1/bocini"
os.environ['RESULTCACHING_HOME'] = custom_cache_directory
os.environ['RESULTCACHING_DISABLE'] = '0'

identifier = 'alexnet' #'resnet50' 'voneresnet-50-robust'
#identifier = 'voneresnet-50-robust'

# if identifier in ['resnet50-barlow', 'custom_model_cv_18_dagger_408', 'efficientnet-b6', 'ViT_L_32_imagenet1k', 'ViT_L_16_imagenet1k', 'r3m_resnet34', 'r3m_resnet50']:
#         identifier_package_mapping = {'resnet50-barlow': 'resnet_selfsup_submission', 'custom_model_cv_18_dagger_408': 'crossvit_18_dagger_408_finetuned',
#                                       'efficientnet-b6': 'efficientnet_models', 'ViT_L_32_imagenet1k': 'ViT', 'ViT_L_16_imagenet1k': 'ViT',
#                                       'r3m_resnet34': 'r3m_main', 'r3m_resnet50': 'r3m_main'}
#         packagename = identifier_package_mapping[identifier]
#         module = importlib.import_module(f"{packagename}.models.base_models")
#         get_submission_model = getattr(module, "get_model")
#         get_submission_layers = getattr(module, "get_layers")
#         import pdb; pdb.set_trace()
#         basemodel = get_submission_model(identifier)
#         layers = get_submission_layers(identifier)
#         brain_model = ModelCommitment(identifier=identifier, activations_model=basemodel, layers=layers)
# else:
#     model = brain_translated_pool[identifier]


# model = ModelCommitment(identifier='my-model', activations_model=model,
#                         # specify layers to consider
#                         layers=LAYERS)

# the region map layer above is not working so I manually changed it inside:
# /Users/ernestobocini/miniconda3/envs/brainscore/lib/python3.7/site-packages/model_tools/brain_transformation/__init__.py

model = brain_translated_pool[identifier]

benchmark = OOD_BehavioralBenchmark()
score = benchmark(model)

# score = score_model(model_identifier=identifier, model=model, benchmark_identifier='bocini-nsd-2023.whole_brain-pls')
import pdb; pdb.set_trace()
print(score)

