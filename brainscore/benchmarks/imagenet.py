from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import brainscore
from brainio_base.stimuli import StimulusSet
from brainio_collection.fetch import fetch_file
from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Score
from brainscore.metrics.accuracy import Accuracy
from brainscore.model_interface import BrainModel

NUMBER_OF_TRIALS = 10


class Imagenet2012(BenchmarkBase):
    def __init__(self):
        self._stimuli_train = self._load_train()
        self._stimuli_val = brainscore.get_stimulus_set('fei-fei.Deng2009.val')
        self._similarity_metric = Accuracy()
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(Imagenet2012, self).__init__(identifier='fei-fei.Deng2009-top1', version=2,
                                           ceiling_func=lambda: ceiling,
                                           parent='ImageNet',
                                           bibtex="""@INPROCEEDINGS{5206848,  
                                                author={J. {Deng} and W. {Dong} and R. {Socher} and L. {Li} and  {Kai Li} and  {Li Fei-Fei}},  
                                                booktitle={2009 IEEE Conference on Computer Vision and Pattern Recognition},   
                                                title={ImageNet: A large-scale hierarchical image database},   
                                                year={2009},  
                                                volume={},  
                                                number={},  
                                                pages={248-255},
                                                url = {https://ieeexplore.ieee.org/document/5206848}
                                            }""")

    def __call__(self, candidate):
        # The proper `fitting_stimuli` to pass to the candidate would be the imagenet training set.
        # For now, since almost all models in our hands were trained with imagenet, we'll just short-cut this
        # by telling the candidate to use its pre-trained imagenet weights.
        candidate.start_task(BrainModel.Task.label, fitting_stimuli=self._stimuli_train)
        stimuli_val = self._stimuli_val[list(set(self._stimuli_val.columns) - {'synset'})]  # do not show label
        predictions = candidate.look_at(stimuli_val, number_of_trials=NUMBER_OF_TRIALS)
        score = self._similarity_metric(
            predictions.sortby('filename'),
            self._stimuli_val.sort_values('filename')['synset'].values
        )
        return score

    def _load_train(self, check_images=False):
        # ImageNet training set is too big to store on and retrieve from S3 (140G).
        # Instead, we store the metadata (csv) on S3 and point to filepaths locally available on our evaluation server.
        csv_path = fetch_file(location_type='S3',
                              location='https://brainio.contrib.s3.amazonaws.com/image_fei-fei_Deng2009_train.csv',
                              sha1='be793cd12a4e3ccffe17a2499e99d2125c1db4c5')  # from packaging
        stimulus_set = pd.read_csv(csv_path)
        stimulus_set = StimulusSet(stimulus_set)
        stimulus_set.identifier = 'fei-fei.Deng2009.train'
        # Link to local files
        stimuli_directory = Path('/braintree/data2/active/common/imagenet_raw/train/')
        stimulus_set['filepath'] = str(stimuli_directory) + '/' + stimulus_set['relative_path']  # vectorized for speed
        stimulus_set.image_paths = dict(zip(stimulus_set['image_id'], stimulus_set['filepath']))
        if check_images:
            # Don't typically run this because it takes about 4 minutes.
            # We could additionally check SHA1 hashes but it would take even longer.
            missing_images = [image_id for image_id in tqdm(stimulus_set['image_id'], desc='check for missing images')
                              if not Path(stimulus_set.get_image(image_id)).is_file()]
            assert len(missing_images) == 0, f"Missing {len(missing_images)} images: {missing_images}"
        stimulus_set['image_label'] = stimulus_set['synset']
        return stimulus_set
