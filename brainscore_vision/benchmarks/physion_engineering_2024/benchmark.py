import numpy as np
import os
import json
import pandas as pd

from brainio.assemblies import BehavioralAssembly
from brainscore_core import Score
from brainscore_vision import load_metric, load_stimulus_set
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.model_interface import BrainModel


class PhysionGlobalDetectionAccuracy(BenchmarkBase):
    def __init__(self):
        # need to download data from s3: videos + json
        
        #self._stimulus_set = json.load(open(os.path.join(os.path.dirname(__file__),
        #                                                'physion_full_brainscore.json'),
        #                              'r'))
        self._stimulus_set  = load_stimulus_set("PhysionOCPSmall")#PhysionGlobalDetection2024")
        # at what degree visual angle stimuli were presented
        self._visual_degrees = 8
        self._similarity_metric = load_metric('accuracy')
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(PhysionGlobalPredictionAccuracy, self).__init__(identifier='Physionv1.5-ocd', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='Physion Engineering',
                                           bibtex="""@article{bear2021physion,
                                               title={Physion: Evaluating physical prediction from vision in humans and machines},
                                               author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
                                               journal={arXiv preprint arXiv:2106.08261},
                                               year={2021}
                                                    }""")

    def __call__(self, candidate):
        # prepare fitting stimuli
        fitting_stimuli = self._stimulus_set[self._stimulus_set['train'] == 1]
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        # prepare test stimuli
        test_stimuli = self._stimulus_set[self._stimulus_set['train'] == 0]
        test_stimuli = place_on_screen(test_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        
        candidate.start_task(BrainModel.Task.video_readout, fitting_stimuli)
        predictions = candidate.look_at(test_stimuli)
        score = self._similarity_metric(
            predictions['choice'],
            test_stimuli['label']
        )
        return score

class PhysionGlobalPredictionAccuracy(BenchmarkBase):
    def __init__(self):
        #self._stimulus_set = json.load(open(os.path.join(os.path.dirname(__file__),
        #                                                'physion_pred_brainscore.json'),
        #                              'r'))
        
        self._stimulus_set  = load_stimulus_set("PhysionGlobalPrediction2024")
        # at what degree visual angle stimuli were presented
        self._visual_degrees = 8
        self._similarity_metric = load_metric('accuracy')
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(PhysionGlobalPredictionAccuracy, self).__init__(identifier='Physionv1.5-ocp', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='Physion Engineering',
                                           bibtex="""@article{bear2021physion,
                                               title={Physion: Evaluating physical prediction from vision in humans and machines},
                                               author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
                                               journal={arXiv preprint arXiv:2106.08261},
                                               year={2021}
                                                    }""")

    def __call__(self, candidate):
        # prepare fitting stimuli
        fitting_stimuli = self._stimulus_set[self._stimulus_set['train'] == 1]
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        # prepare test stimuli
        test_stimuli = self._stimulus_set[self._stimulus_set['train'] == 0]
        test_stimuli = place_on_screen(test_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        
        candidate.start_task(BrainModel.Task.video_readout, fitting_stimuli)
        predictions = candidate.look_at(test_stimuli)
        score = self._similarity_metric(
            predictions['choice'],
            test_stimuli['label']
        )
        return score


class PhysionGlobalDetectionIntraScenarioAccuracy(BenchmarkBase):
    def __init__(self):
        #self._stimulus_set = json.load(open(os.path.join(os.path.dirname(__file__),
        #                                                'physion_full_intra_scenario_brainscore.json'),
        #                              'r'))
        self._stimulus_set  = load_stimulus_set("PhysionGlobalDetectionIntraScenario2024")
        # at what degree visual angle stimuli were presented
        self._visual_degrees = 8
        self._similarity_metric = load_metric('accuracy')
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(PhysionGlobalPredictionAccuracy, self).__init__(identifier='Physionv1.5-ocd-intra-generalization', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='Physion Engineering',
                                           bibtex="""@article{bear2021physion,
                                               title={Physion: Evaluating physical prediction from vision in humans and machines},
                                               author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
                                               journal={arXiv preprint arXiv:2106.08261},
                                               year={2021}
                                                    }""")

    def __call__(self, candidate):
        # prepare fitting stimuli
        fitting_stimuli = self._stimulus_set[self._stimulus_set['train'] == 1]
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        # prepare test stimuli
        test_stimuli = self._stimulus_set[self._stimulus_set['train'] == 0]
        test_stimuli = place_on_screen(test_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        
        candidate.start_task(BrainModel.Task.video_readout, fitting_stimuli)
        predictions = candidate.look_at(test_stimuli)
        score = self._similarity_metric(
            predictions['choice'],
            test_stimuli['label']
        )
        return score

class PhysionGlobalPredictionIntraScenarioAccuracy(BenchmarkBase):
    def __init__(self):
        #self._stimulus_set = json.load(open(os.path.join(os.path.dirname(__file__),
        #                                                'physion_pred_intra_scenario_brainscore.json'),
        #                              'r'))
        self._stimulus_set  = load_stimulus_set("PhysionGlobalPredictionIntraScenario2024")
        # at what degree visual angle stimuli were presented
        self._visual_degrees = 8
        self._similarity_metric = load_metric('accuracy')
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(PhysionGlobalPredictionAccuracy, self).__init__(identifier='Physionv1.5-ocp-intra-generalization', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='Physion Engineering',
                                           bibtex="""@article{bear2021physion,
                                               title={Physion: Evaluating physical prediction from vision in humans and machines},
                                               author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
                                               journal={arXiv preprint arXiv:2106.08261},
                                               year={2021}
                                                    }""")

    def __call__(self, candidate):
        # prepare fitting stimuli
        fitting_stimuli = self._stimulus_set[self._stimulus_set['train'] == 1]
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        # prepare test stimuli
        test_stimuli = self._stimulus_set[self._stimulus_set['train'] == 0]
        test_stimuli = place_on_screen(test_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        
        candidate.start_task(BrainModel.Task.video_readout, fitting_stimuli)
        predictions = candidate.look_at(test_stimuli)
        score = self._similarity_metric(
            predictions['choice'],
            test_stimuli['label']
        )
        return score


class PhysionSnippetPredictionAccuracy(BenchmarkBase):
    def __init__(self):
        #self._stimulus_set = json.load(open(os.path.join(os.path.dirname(__file__),
        #                                                'physion_behavior_brainscore.json')),
        #                              'r')

        self._stimulus_set  = load_stimulus_set("PhysionSnippetPrediction2024")
        # at what degree visual angle stimuli were presented
        self._visual_degrees = 8
        self._similarity_metric = load_metric('accuracy')
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(PhysionSnippetPredictionAccuracy, self).__init__(identifier='Physionv1.5-snippet-simulation-performance', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='PhysionV1.5',
                                           bibtex="""@article{bear2021physion,
                                               title={Physion: Evaluating physical prediction from vision in humans and machines},
                                               author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
                                               journal={arXiv preprint arXiv:2106.08261},
                                               year={2021}
                                                    }""")

    def __call__(self, candidate):
        # prepare fitting stimuli
        fitting_stimuli = self._stimulus_set[self._stimulus_set['train'] == 1]
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        # prepare test stimuli
        test_stimuli = self._stimulus_set[self._stimulus_set['train'] == 0]
        test_stimuli = place_on_screen(test_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        
        candidate.start_task(BrainModel.Task.video_readout, fitting_stimuli)
        predictions = candidate.look_at(test_stimuli)
        predictions = aggregate_preds(predictions)
        score = self._similarity_metric(
            predictions['choice'],
            test_stimuli['label']
        )
        return score

class PhysionSnippetDetectionAccuracy(BenchmarkBase):
    def __init__(self):
        #self._stimulus_set = json.load(open(os.path.join(os.path.dirname(__file__),
        #                                                'physion_behavior_brainscore.json')),
        #                              'r')
        self._stimulus_set  = load_stimulus_set("PhysionSnippetDetection2024")
        # at what degree visual angle stimuli were presented
        self._visual_degrees = 8
        self._similarity_metric = load_metric('accuracy')
        ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        super(PhysionSnippetDetectionAccuracy, self).__init__(identifier='Physionv1.5-snippet-rollout-performance', version=1,
                                           ceiling_func=lambda: ceiling,
                                           parent='PhysionV1.5',
                                           bibtex="""@article{bear2021physion,
                                               title={Physion: Evaluating physical prediction from vision in humans and machines},
                                               author={Bear, Daniel M and Wang, Elias and Mrowca, Damian and Binder, Felix J and Tung, Hsiao-Yu Fish and Pramod, RT and Holdaway, Cameron and Tao, Sirui and Smith, Kevin and Sun, Fan-Yun and others},
                                               journal={arXiv preprint arXiv:2106.08261},
                                               year={2021}
                                                    }""")

    def __call__(self, candidate):
        # prepare fitting stimuli
        fitting_stimuli = self._stimulus_set[self._stimulus_set['train'] == 1]
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        # prepare test stimuli
        test_stimuli = self._stimulus_set[self._stimulus_set['train'] == 0]
        test_stimuli = place_on_screen(test_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        
        candidate.start_task(BrainModel.Task.video_readout, fitting_stimuli)
        predictions = candidate.look_at(test_stimuli)
        predictions = aggregate_preds(predictions)
        score = self._similarity_metric(
            predictions['choice'],
            test_stimuli['label']
        )
        return score

def aggregate_preds(predictions):
    # Extract stimulus_path and choice from proba
    stimulus_paths = predictions['stimulus_path'].values
    choices = predictions['choice'].values
    
    # Extract video names from stimulus paths
    video_names = np.array([path.split('_img_snippet_')[0] for path in stimulus_paths])
    
    # Create a DataFrame for easier manipulation
    data = pd.DataFrame({'video_name': video_names, 'choice': choices})
    
    # Group by video name and aggregate choices
    aggregated = data.groupby('video_name')['choice'].agg(lambda x: 1 if any(x) else 0).reset_index()
    
    # Create a mapping from video name to aggregated choice
    video_to_choice = dict(zip(aggregated['video_name'], aggregated['choice']))
    
    # Apply the aggregated choice back to the original data
    aggregated_choices = [video_to_choice[video] for video in video_names]
    
    # If you need to modify the BehavioralAssembly object, you can recreate it as needed
    predictions = BehavioralAssembly(aggregated_choices,
                               coords={
                                   'stimulus_path': ('presentation', video_names),
                               },
                               dims=['presentation'])
    return predictions


