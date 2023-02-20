import os

ROOT_REPO_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
WORD_AND_SPEAKER_ENCODINGS_PATH = os.path.join(ROOT_REPO_DIR, 'robustness', 'audio_functions', 'word_and_speaker_encodings_jsinv3.pckl')
ASSETS_PATH = os.path.join(ROOT_REPO_DIR, 'assets')
WORDNET_ID_TO_HUMAN_PATH = os.path.join(ROOT_REPO_DIR, 'analysis_scripts', 'wordnetID_to_human_identifier.txt')
IMAGENET_PATH = None
JSIN_PATH = None

# fMRI dataset paths
fMRI_DATA_PATH = os.path.join(ASSETS_PATH, 'fMRI_natsound_data')
