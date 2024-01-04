from base_models import get_model, get_layers
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment

def test_eval():
    model_name = 'alexnet_l2_3_robust'
    benchmark_name = 'dietterich.Hendrycks2019-noise-top1'
    model=get_model(model_name)
    layers=get_layers(model_name)
    
    model = ModelCommitment(identifier=model_name, activations_model=model, layers=layers)
    
    score = score_model(model_identifier=model_name, model=model, benchmark_identifier=benchmark_name)
    print('Average correct for %s on %s: %f'%(model_name, benchmark_name, score.raw.mean()))

if __name__ == "__main__":
    test_eval()
