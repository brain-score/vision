import os

import numpy as np
import pandas as pd

from brainio_base.assemblies import NeuroidAssembly
from brainio_base.stimuli import StimulusSet
from brainscore.metrics.pca import PCA


class TestPCA:
    def test_single_batch(self):
        rng = np.random.RandomState(0)

        def model(stimuli, num_units=30):
            values = rng.rand(len(stimuli), num_units)
            assembly = NeuroidAssembly(values, coords={
                'image_id': ('presentation', stimuli['image_id'] if isinstance(stimuli, pd.DataFrame) else stimuli),
                'image_meta': ('presentation', stimuli['image_id'] if isinstance(stimuli, pd.DataFrame) else stimuli),
                'neuroid_id': ('neuroid', list(range(num_units))),
                'layer': ('neuroid', ['dummy-layer'] * num_units),
            }, dims=['presentation', 'neuroid'])
            return assembly

        stimuli = StimulusSet({'image_id': ['1', '2', '3'], 'degrees': [4, 4, 4]})
        local_dir = os.path.join(os.path.dirname(__file__))
        stimuli.image_paths = {'1': os.path.join(local_dir, 'image_000001.png'),
                               '2': os.path.join(local_dir, 'image_000002.png'),
                               '3': os.path.join(local_dir, 'image_000003.png'),
                               }
        stimuli.name = 'test3'

        pca = PCA(n_components=10)
        pca_assembly = pca(model_identifier='test', model=model, stimuli=stimuli)
        assert len(pca_assembly['presentation']) == 3
        assert len(pca_assembly['neuroid']) == 10

    def test_three_batch(self):
        rng = np.random.RandomState(0)

        def model(stimuli, num_units=30):
            values = rng.rand(len(stimuli), num_units)
            assembly = NeuroidAssembly(values, coords={
                'image_id': ('presentation', stimuli['image_id'] if isinstance(stimuli, pd.DataFrame) else stimuli),
                'image_meta': ('presentation', stimuli['image_id'] if isinstance(stimuli, pd.DataFrame) else stimuli),
                'neuroid_id': ('neuroid', list(range(num_units))),
                'layer': ('neuroid', ['dummy-layer'] * num_units),
            }, dims=['presentation', 'neuroid'])
            return assembly

        stimuli = StimulusSet({'image_id': ['0', '1', '2', '3', '4'], 'degrees': 4})
        local_dir = os.path.join(os.path.dirname(__file__))
        stimuli.image_paths = {'0': os.path.join(local_dir, 'image_000000.png'),
                               '1': os.path.join(local_dir, 'image_000001.png'),
                               '2': os.path.join(local_dir, 'image_000002.png'),
                               '3': os.path.join(local_dir, 'image_000003.png'),
                               '4': os.path.join(local_dir, 'image_000004.png'),
                               }
        stimuli.name = 'test5'

        pca = PCA(n_components=10)
        pca_assembly = pca(model_identifier='test', model=model, stimuli=stimuli, batch_size=2)
        assert len(pca_assembly['presentation']) == 5
        assert len(pca_assembly['neuroid']) == 10
