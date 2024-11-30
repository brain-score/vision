import subprocess
import torch
from pathlib import Path
from torchvision.models import resnet18
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from collections import OrderedDict
import functools

# Define preprocessing
preprocessing = functools.partial(load_preprocess_images, image_size=224)

def run_download_script(file_url, destination_path):
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    script_path = script_dir / "download_google_drive.sh"
    
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"Running script: {script_path}")
    result = subprocess.run(
        [str(script_path), file_url, destination_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Script failed with return code {result.returncode}:\n{result.stderr}")
    print("Download completed successfully.")


# Custom model loader
def get_model(name):
    assert name == 'barlow_twins_custom'

    # File URL, script path, and destination path for the checkpoint
    file_url = "https://drive.google.com/uc?export=download&id=16j13GkdftLYHNGutKeP2LWhMGcTIis5n"
    script_path = "./download_google_drive.sh"
    checkpoint_path = Path("./barlow_twins-custom_dataset_3-685qxt9j-ep=399.ckpt")

    # Download the checkpoint using the shell script if it doesn't exist
    if not checkpoint_path.exists():
        print("Checkpoint not found. Downloading...")
        run_download_script(file_url, str(checkpoint_path))

    # Validate checkpoint file
    if checkpoint_path.exists() and checkpoint_path.stat().st_size < 1e6:  # Adjust size threshold if needed
        raise ValueError(f"Downloaded checkpoint seems invalid. File size: {checkpoint_path.stat().st_size} bytes")

    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint. Error: {e}")

    # Fix state_dict by removing 'backbone.' prefix
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):  # Remove 'backbone.' prefix
            new_key = k.replace('backbone.', '')
            new_state_dict[new_key] = v
      
    # Load the modified state_dict into the model
    model = resnet18(pretrained=False)
    model.load_state_dict(new_state_dict, strict=False)
    print(model)
    
    # Wrap the model for Brain-Score
    activations_model = PytorchWrapper(identifier='barlow_twins_custom', model=model, preprocessing=preprocessing)
    return ModelCommitment(
        identifier='barlow_twins_custom',
        activations_model=activations_model,
        layers=['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    )

def get_model_list():
    return ['barlow_twins_custom']

# Specify layers to test
def get_layers(name):
    assert name == 'barlow_twins_custom'
    return ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

# Optional: Add a BibTeX reference
def get_bibtex(model_identifier):
    return """
@article{your_barlow_twins_reference,
  title={Barlow Twins Trained on Custom Dataset},
  author={Claudia Noche},
  year={2024},
"""

if __name__ == '__main__':
    from brainscore_vision.model_helpers.check_submission import check_models
    check_models.check_base_models(__name__)
