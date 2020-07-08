from brainscore import score_model
from candidate_models import brain_translated_pool


def run_benchmark(benchmark_identifier, model_name):
    print(f'>>>>>Start running model {model_name} on benchmark {benchmark_identifier}')
    model = brain_translated_pool[model_name]
    score = score_model(model_identifier=model_name, model=model, benchmark_identifier=benchmark_identifier)
    regions = model.layer_model._layer_model.recorded_regions
    return score


if __name__ == '__main__':
    run_benchmark('dicarlo.Majaj2015.IT-pls','alexnet')