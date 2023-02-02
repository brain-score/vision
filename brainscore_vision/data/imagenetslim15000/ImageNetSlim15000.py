import os
import numpy as np
import pandas as pd
from brainio_base.stimuli import StimulusSet
from brainio_collection.packaging import package_stimulus_set


def collect_stimuli(data_path):
    assert os.path.isdir(data_path)
    stimulus_df = pd.read_pickle(os.path.join(data_path,'info.pkl'))
    stimulus_set = StimulusSet(stimulus_df)
    stimulus_set.image_paths = {
        stimulus_set.at[idx, 'image_id']: os.path.join(data_path,'data',stimulus_set.at[idx, 'image_name']) for idx in range(len(stimulus_set))}
    return stimulus_set


def main():
    stimulus_path = os.path.normpath('/Users/chongguo/Dropbox (MIT)/CG@DiCarlo/Datasets/ImageNetSlim15000')
    stimulus_set = collect_stimuli(stimulus_path)
    stimulus_set.identifier = 'dicarlo.ImageNetSlim15000'
    package(stimulus_set)


def package(stimulus_set):
    print("Packaging stimuli")
    package_stimulus_set(stimulus_set, stimulus_set_identifier=stimulus_set.identifier, bucket_name= 'brainio.requested')


if __name__ == '__main__':
    main()
