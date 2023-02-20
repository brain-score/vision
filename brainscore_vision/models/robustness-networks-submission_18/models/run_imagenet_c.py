from base_models import get_model, get_layers
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment
import pickle as pckl
import argparse
import os

SAVE_DIR = '/om2/user/jfeather/projects/brain_score/robustness-networks-submission/results/imagenet-c-results'

def test_eval_model(model_name):
#     model_name = 'alexnet_l2_3_robust'
    benchmark_list = ['dietterich.Hendrycks2019-noise-top1', 'dietterich.Hendrycks2019-blur-top1',
                      'dietterich.Hendrycks2019-weather-top1', 'dietterich.Hendrycks2019-digital-top1']
    benchmark_scores = {}

    for benchmark_name in benchmark_list:
        model=get_model(model_name)
        layers=get_layers(model_name)
        
        model = ModelCommitment(identifier=model_name, activations_model=model, layers=layers)
        
        score = score_model(model_identifier=model_name, model=model, benchmark_identifier=benchmark_name)
        print('Average correct for %s on %s: %f'%(model_name, benchmark_name, score.raw.mean()))
        benchmark_scores[benchmark_name] = score.raw.mean()

    return benchmark_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model_name', type=str, help='name of the model to run')

    args = parser.parse_args()

    if os.path.isfile(os.path.join(SAVE_DIR, args.model_name + '-imagenetc.pckl')):
        raise ValueError('File %s exists and not forcing overwriting'%os.path.join(SAVE_DIR, args.model_name + '-imagenetc.pckl'))

    benchmark_scores = test_eval_model(args.model_name)
    
    pckl.dump(benchmark_scores, open(os.path.join(SAVE_DIR, args.model_name + '-imagenetc.pckl'), 'wb'))
