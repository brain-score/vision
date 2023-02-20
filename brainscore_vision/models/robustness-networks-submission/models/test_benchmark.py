


from base_model import get_model, get_layers


model_name = 'alexnet_l2_3_robust'
model=get_model(model_name)
layers=get_layers(model_name)

from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment

model = ModelCommitment(identifier=model_name, activations_model=model, layers=layers)

score = score_model(model_identifier=model_name, model=model, benchmark_identifier='dietterich.Hendrycks2019-noise-top1')
print(score)

