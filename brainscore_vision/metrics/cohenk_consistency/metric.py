import sklearn.metrics
import numpy as np

from brainio.assemblies import walk_coords
from brainscore_vision.metrics import Score


class ModelHumanCohenK:
    def __call__(self, model_data, human_data):
        return self.get_score(model_data, human_data)

    def get_score(self, MD, HD):
        cohen_ks = []
        scenarios = sorted(set(MD['scenario'].values))
    
        for scenario in scenarios:
            # Get subsets for the current scenario for coord, dims, value in walk_coords(X)
            _MD = {k: [v[i] for i in range(len(v)) if MD['scenario'][i] == scenario] for k, _, v in walk_coords(MD)}
            _HD = {k: [v[i] for i in range(len(v)) if HD['scenario'][i] == scenario] for k, _, v in walk_coords(HD)}
            _MD['stimulus_id'] = [f[:-8] for f in _MD['stimulus_id']]

            # Sort by 'stimulus_id'
            sort_idx_MD = sorted(range(len(_MD['stimulus_id'])), key=lambda i: _MD['stimulus_id'][i])
            _MD = {k: [v[i] for i in sort_idx_MD] for k, v in _MD.items()}
    
            measures_for_model = []
    
            gameIDs = set(_HD['gameID'])
            for gameID in gameIDs:
                # Get one game
                _HD_game = {k: [v[i] for i in range(len(v)) if _HD['gameID'][i] == gameID] for k, v in _HD.items()}
    
                # Sort by 'stimulus_id'
                sort_idx_HD_game = sorted(range(len(_HD_game['stimulus_id'])), key=lambda i: _HD_game['stimulus_id'][i])
                _HD_game = {k: [v[i] for i in sort_idx_HD_game] for k, v in _HD_game.items()}
    
                # Find common stimulus IDs
                human_stim_names = set(_HD_game['stimulus_id'])
                model_stim_names = set(_MD['stimulus_id'])
                joint_stim_names = human_stim_names.intersection(model_stim_names)
    
                # Subset both models to ensure only common stims are used
                _MD_common = {k: [v[i] for i in range(len(v)) if _MD['stimulus_id'][i] in joint_stim_names] for k, v in _MD.items()}
                _HD_game_common = {k: [v[i] for i in range(len(v)) if _HD_game['stimulus_id'][i] in joint_stim_names] for k, v in _HD_game.items()}
    
                # Pull response vector
                human_responses = np.array(_HD_game_common['responseBool'])  # Get human response and cast to int
                model_responses = np.array(_MD_common['choice'])
    
                # Add similarity metric
                measure = sklearn.metrics.cohen_kappa_score(model_responses, human_responses)
                measures_for_model.append(measure)
    
            # Get percentiles over the range of measures
            med = np.percentile(measures_for_model, 50)
            cohen_ks += [med]
    
        cohen_ks = np.array(cohen_ks)
        center = np.mean(cohen_ks)
        error = np.std(cohen_ks)
    
        score = Score(center)
        score.attrs['error'] = error
        score.attrs[Score.RAW_VALUES_KEY] = cohen_ks
        return score
