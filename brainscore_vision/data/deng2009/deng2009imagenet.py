import numpy as np
import os
import pandas as pd
from pathlib import Path

from brainio_base.stimuli import StimulusSet
from brainio_collection.packaging import package_stimulus_set


def collect_stimuli(data_dir):
    stimulus_set = pd.read_csv(data_dir / 'imagenet2012.csv')
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.image_paths = {row.image_id: row.filepath for row in stimulus_set.itertuples()}
    stimulus_set['image_path_within_store'] = stimulus_set['filename'].apply(
        lambda filename: os.path.splitext(filename)[0])
    stimulus_set = stimulus_set[['image_id', 'label', 'synset', 'image_file_sha1', 'image_path_within_store']]
    assert len(np.unique(stimulus_set['image_id'])) == len(stimulus_set), "duplicate entries"
    return stimulus_set


def main():
    data_dir = Path('/braintree/home/msch/brain-score/brainscore/benchmarks')
    assert os.path.isdir(data_dir)

    stimuli = collect_stimuli(data_dir)
    stimuli.identifier = 'imagenet_val'

    print('Packaging stimuli')
    package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.identifier, bucket_name='brainio.contrib')


if __name__ == '__main__':
    main()
