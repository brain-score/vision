import numpy as np
import os
import json
import pandas as pd

from brainio.assemblies import BehavioralAssembly
from brainscore_core import Score
from brainscore_vision import load_metric, load_stimulus_set, load_dataset
from brainscore_vision.benchmarks import BenchmarkBase
from brainscore_vision.benchmark_helpers.screen import place_on_screen
from brainscore_vision.model_interface import BrainModel
from brainscore_vision.utils import LazyLoad


class PhysionGlobalDetectionHumanCohenK(BenchmarkBase):
    def __init__(self):
        self._stimulus_set  = load_stimulus_set("PhysionGlobalDetection2024")
        self._human_data = LazyLoad(lambda: load_dataset("PhysionHumanDetection2024"))
        self._visual_degrees = 8
        self._similarity_metric = load_metric('cohenk_consistency')
        #ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        ceiling = Score(1)
        super(PhysionGlobalDetectionHumanCohenK, self).__init__(identifier='Physionv1.5-ocd-cohenk', version=1,
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
            predictions,
            self._human_data,
        )
        return score

class PhysionGlobalPredictionHumanCohenK(BenchmarkBase):
    def __init__(self):
        self._stimulus_set  = load_stimulus_set("PhysionGlobalPrediction2024")
        self._human_data = LazyLoad(lambda: load_dataset("PhysionHumanPrediction2024"))
        self._visual_degrees = 8
        self._similarity_metric = load_metric('cohenk_consistency')
        #ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        ceiling = Score(1)
        super(PhysionGlobalPredictionHumanCohenK, self).__init__(identifier='Physionv1.5-ocp-cohenk', version=1,
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
            predictions,
            self._human_data,
        )
        return score


class PhysionGlobalDetectionIntraScenarioHumanCohenK(BenchmarkBase):
    def __init__(self):
        self._stimulus_set  = load_stimulus_set("PhysionGlobalDetection2024")
        self._human_data = LazyLoad(lambda: load_dataset("PhysionHumanDetection2024"))
        self._visual_degrees = 8
        self._similarity_metric = load_metric('cohenk_consistency')
        #ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        ceiling = Score(1)
        super(PhysionGlobalDetectionIntraScenarioHumanCohenK, self).__init__(identifier='Physionv1.5-ocd-intra-generalization-cohenk', version=1,
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
        fitting_stimuli = fitting_stimuli[fitting_stimuli['intra_generalizability'] == 1]
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        # prepare test stimuli
        test_stimuli = self._stimulus_set[self._stimulus_set['train'] == 0]
        test_stimuli = test_stimuli[test_stimuli['intra_generalizability'] == 1]
        test_stimuli = place_on_screen(test_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        
        candidate.start_task(BrainModel.Task.video_readout, fitting_stimuli)
        predictions = candidate.look_at(test_stimuli)
        score = self._similarity_metric(
            predictions,
            self._human_data,
        )
        return score

class PhysionGlobalPredictionIntraScenarioHumanCohenK(BenchmarkBase):
    def __init__(self):
        self._stimulus_set  = load_stimulus_set("PhysionGlobalPrediction2024")
        self._human_data = LazyLoad(lambda: load_dataset("PhysionHumanPrediction2024"))
        self._visual_degrees = 8
        self._similarity_metric = load_metric('cohenk_consistency')
        #ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        ceiling = Score(1)
        super(PhysionGlobalPredictionIntraScenarioHumanCohenK, self).__init__(identifier='Physionv1.5-ocp-intra-generalization-cohenk', version=1,
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
        fitting_stimuli = fitting_stimuli[fitting_stimuli['intra_generalizability'] == 1]
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        # prepare test stimuli
        test_stimuli = self._stimulus_set[self._stimulus_set['train'] == 0]
        test_stimuli = test_stimuli[test_stimuli['intra_generalizability'] == 1]
        test_stimuli = place_on_screen(test_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        
        candidate.start_task(BrainModel.Task.video_readout, fitting_stimuli)
        predictions = candidate.look_at(test_stimuli)
        score = self._similarity_metric(
            predictions,
            self._human_data,
        )
        return score

class PhysionSnippetDetectionHumanCohenK(BenchmarkBase):
    def __init__(self):
        self._stimulus_set  = load_stimulus_set("PhysionSnippetDetection2024")
        self._human_data = LazyLoad(lambda: load_dataset("PhysionHumanDetection2024"))
        self._visual_degrees = 8
        self._similarity_metric = load_metric('cohenk_consistency')
        #ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        ceiling = Score(1)
        super(PhysionSnippetDetectionHumanCohenK, self).__init__(identifier='Physionv1.5-snippet-rollout-performance-cohenk', version=1,
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
        test_stimuli_assembly = BehavioralAssembly(test_stimuli['label'].values,
                               coords=
                               {'stimulus_id': ('presentation', test_stimuli['stimulus_id'].values),
                                'choice': ('presentation', test_stimuli['label'].values),
                                'scenario': ('presentation', test_stimuli['scenario'].values)},
                               dims=['presentation'])
        test_stimuli_assembly = aggregate_preds(test_stimuli_assembly)
        score = self._similarity_metric(
            predictions,
            test_stimuli_assembly
        )
        return score

class PhysionSnippetDetectionIntraScenarioHumanCohenK(BenchmarkBase):
    def __init__(self):
        self._stimulus_set  = load_stimulus_set("PhysionSnippetDetection2024")
        self._human_data = LazyLoad(lambda: load_dataset("PhysionHumanDetection2024"))
        self._visual_degrees = 8
        self._similarity_metric = load_metric('cohenk_consistency')
        #ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        ceiling = Score(1)
        super(PhysionSnippetDetectionIntraScenarioHumanCohenK, self).__init__(identifier='Physionv1.5-snippet-rollout-intra-performance-cohenk', version=1,
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
        fitting_stimuli = fitting_stimuli[fitting_stimuli['intra_generalizability'] == 1]
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        # prepare test stimuli
        test_stimuli = self._stimulus_set[self._stimulus_set['train'] == 0]
        test_stimuli = test_stimuli[test_stimuli['intra_generalizability'] == 1]
        test_stimuli = place_on_screen(test_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        
        candidate.start_task(BrainModel.Task.video_readout, fitting_stimuli)
        predictions = candidate.look_at(test_stimuli)
        predictions = aggregate_preds(predictions)
        test_stimuli_assembly = BehavioralAssembly(test_stimuli['label'].values,
                               coords=
                               {'stimulus_id': ('presentation', test_stimuli['stimulus_id'].values),
                                'choice': ('presentation', test_stimuli['label'].values),
                                'scenario': ('presentation', test_stimuli['scenario'].values)},
                               dims=['presentation'])
        test_stimuli_assembly = aggregate_preds(test_stimuli_assembly)
        score = self._similarity_metric(
            predictions,
            test_stimuli_assembly
        )
        return score

class PhysionSnippetSimulationHumanCohenK(BenchmarkBase):
    def __init__(self):
        self._stimulus_set  = load_stimulus_set("PhysionSnippetDetection2024")
        self._test_set  = load_stimulus_set("PhysionGlobalPrediction2024")
        self._human_data = LazyLoad(lambda: load_dataset("PhysionHumanDetection2024"))
        self._visual_degrees = 8
        self._similarity_metric = load_metric('cohenk_consistency')
        #ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
        ceiling = Score(1)
        super(PhysionSnippetDetectionHumanCohenK, self).__init__(identifier='Physionv1.5-snippet-simulation-cohenk', version=1,
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
        test_stimuli = self._test_set[self._test_set['train'] == 0]
        test_stimuli = place_on_screen(test_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                          source_visual_degrees=self._visual_degrees)
        
        candidate.start_task(BrainModel.Task.video_readout, fitting_stimuli)
        simulated_feats = candidate.look_at(test_stimuli, simulation=True)
        predictions = aggregate_preds(simulated_feats, test_stimuli['stimulus_id'].values)
        test_stimuli_assembly = BehavioralAssembly(test_stimuli['label'].values,
                               coords=
                               {'stimulus_id': ('presentation', test_stimuli['stimulus_id'].values),
                                'choice': ('presentation', test_stimuli['label'].values),
                                'scenario': ('presentation', test_stimuli['scenario'].values)},
                               dims=['presentation'])
        test_stimuli_assembly = aggregate_preds(test_stimuli_assembly)
        score = self._similarity_metric(
            predictions,
            test_stimuli_assembly
        )
        return score

def aggregate_preds(predictions, ordered_stimuli=None):
    # Extract stimulus_path and choice from proba
    stimulus_paths = predictions['stimulus_id'].values
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
    if ordered_stimuli is not None:
        aggregated_choices = [video_to_choice[video.split('_img')[0]] for video in ordered_stimuli]
        # If you need to modify the BehavioralAssembly object, you can recreate it as needed
        predictions = BehavioralAssembly(aggregated_choices,
                                   coords={
                                       'stimulus_id': ('presentation', [video for video in ordered_stimuli]),
                                   },
                                   dims=['presentation'])
    else:
        aggregated_choices = [video_to_choice[video] for video in aggregated['video_name']]
        # If you need to modify the BehavioralAssembly object, you can recreate it as needed
        predictions = BehavioralAssembly(aggregated_choices,
                                   coords={
                                       'stimulus_id': ('presentation', [video+'.mp4' for video in aggregated['video_name']]),
                                   },
                                   dims=['presentation'])
    
    return predictions

