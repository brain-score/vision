import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)

CHECKPOINT_DIR = '/home/ozhan/TDANN/spacetorch/tdann/checkpoints'
def select_model(model_type:str):
        if model_type == 'CB_SOM_RN_18':
            dir = f'{parent_dir}/trunks/89 epochs with batchsize of 256_step_size =(60, 80) of SOResnet with alpha = 0.01 to 0.008_weight_dc=0_min is 0.003.pt'
        return dir
