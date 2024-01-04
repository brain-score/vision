import torch
from robustness.model_utils import make_and_restore_model, model_dataset_from_store
from robustness.datasets import ImageNet
from pathlib import Path

def load_and_save_model(model_name, file_name):
    ds = ImageNet('/tmp')
    model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
                                      resume_path=f'pretrained_models/{model_name}/{file_name}')
    model = model.model
    model.to(torch.device('cpu'))
    Path(f'extracted_models/{model_name}/').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f'extracted_models/{model_name}/{file_name}')

if __name__ == '__main__':
    models_to_load = {'resnet50_batch512_3steps_eps0.01':['checkpoint.pt.best'],
                      'resnet50_batch512_3steps_eps0.02':['checkpoint.pt.best'],
                      'resnet50_batch512_3steps_eps0.05':['checkpoint.pt.best']}
    for model, checkpoints_to_load in models_to_load.items():
        for checkpoint in checkpoints_to_load:
            load_and_save_model(model, checkpoint)
            print(f"Saved {model}/{checkpoint}")