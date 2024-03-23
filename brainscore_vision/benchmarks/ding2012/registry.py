import os
import pandas as pd
import numpy as np

from brainio.assemblies import BehavioralAssembly
from result_caching import store

# assembly: 
# - stimulus_id = COH_DIRECTION  (0 - RIGHT, 90 - UP, 180- LEFT, 270 DOWN)
# - truth = CORRECT * CHOICE + (1 - CORRECT) * (3 - CHOICE)
# - choice = CHOICE
# - response_time = SAC ONSET - DOTS ONSET
# - repetition = number of trials with the same stimulus_id

data_dir = '/home/ytang/workspace/tmp/data-FEF/data'
video_dir = '/home/ytang/workspace/data/FEF/videos'

@store()
def load_dataset(identifier="Ding2012", filter_directions=[0, 180], num_samples=100):

    record = {}
    stimulus_ids = []
    truths = []
    choices = []
    response_times = []
    repetitions = []
    cohs = []
    corrects = []
    monkey_names = []
    for filename in os.listdir(data_dir):
        csv = pd.read_csv(os.path.join(data_dir, filename))
        monkey_name = filename[:2]
        trial_no_ = None
        for _, row in csv.iterrows():
            trial_no = row['TRIALNO']
            if trial_no == trial_no_: continue
            trial_no_ = trial_no
            coh = row['COH']
            if coh == 0: continue  # skip 0 coherence
            direction = int(row['DIRECTION'])
            if filter_directions:
                if direction not in filter_directions: continue
            correct = int(row['CORRECT'])
            choice = direction if correct else (direction + 180) % 360
            truth = direction
            sac_onset = row['SAC ONSET']
            dots_onset = row['DOTS ONSET']
            response_time = sac_onset - dots_onset
            stimulus_id = f"{coh * 100}_{(direction)}"
            if stimulus_id not in record:
                record[stimulus_id] = 0
            else:
                record[stimulus_id] += 1
            repetition = record[stimulus_id]

            stimulus_ids.append(stimulus_id)
            truths.append(truth)
            choices.append(choice)
            response_times.append(response_time)
            repetitions.append(repetition)
            cohs.append(coh)
            corrects.append(correct)
            monkey_names.append(monkey_name)

    assembly = BehavioralAssembly(
        choices,
        dims=('presentation',),
        coords={
            'stimulus_id': ('presentation', stimulus_ids),
            'truth': ('presentation', truths),
            'choice': ('presentation', choices),
            'response_time': ('presentation', response_times),
            'repetition': ('presentation', repetitions),
            'coh': ('presentation', cohs),
            'correct': ('presentation', corrects),
            "monkey": ("presentation", monkey_names),
        }
    )
    return assembly


@store()
def load_stimulus_set(identifier, filter_directions=[0, 180], num_samples=100):
    # add stimulus_set
    if identifier == "Ding2012.train_stimuli":
        train = True
    elif identifier == "Ding2012.test_stimuli":
        train = False

    coherences = [0.032, 0.064, 0.128, 0.256, 0.512]
    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    if filter_directions:
        directions = filter_directions

    stimulus_ids = []
    stimulus_paths = []
    stimulus_directions = []

    # make num_samples samples for each case

    for i in range(num_samples):
        for coh in coherences:
            for dir in directions:
                sp = f"train_{coh*100}_{dir}_{i}" if train else f"test_{coh*100}_{dir}_{i}"
                stimulus_ids.append(sp)
                stimulus_paths.append(os.path.join(video_dir, f"{sp}.mp4"))
                stimulus_directions.append(dir)


    from brainio.stimuli import StimulusSet
    stimulus_paths = [os.path.join(video_dir, f"{stimulus_id}.mp4") for stimulus_id in stimulus_ids]

    stimulus_set = {}
    stimulus_set["stimulus_id"] = stimulus_ids
    stimulus_set["direction"] = stimulus_directions
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.stimulus_paths = {id:path for id, path in zip(stimulus_set["stimulus_id"], stimulus_paths)}
    stimulus_set.identifier = identifier

    return stimulus_set


if __name__ == "__main__":
    assembly = load_dataset()

    from brainscore.metrics.ceiling import _SplitHalvesConsistency
    from brainscore.metrics import Score
        
    class CoherenceCorrelation:
        def __call__(self, x, y):
            from scipy.stats import pearsonr
            x = x.correct.groupby('coh').mean()
            y = y.correct.groupby('coh').mean()
            score = pearsonr(x.values, y.values)[0]
            score = Score(score)
            return score
        
    class DirectionCorrelation:
        def __call__(self, x, y):
            from scipy.stats import pearsonr
            x = x.correct.groupby('truth').mean()
            y = y.correct.groupby('truth').mean()
            score = pearsonr(x.values, y.values)[0]
            score = Score(score)
            return score
        
    # coh_consistency = _SplitHalvesConsistency(CoherenceCorrelation())
    # dir_consistency = _SplitHalvesConsistency(DirectionCorrelation())
    # print(coh_consistency(assembly))  # array([9.99208981e-01, 2.07448067e-04])
    # print(dir_consistency(assembly))  # array([0.97989633, 0.00381708])

    print(assembly.correct.groupby('coh').mean())
    # array([0.65296804, 0.75795455, 0.9       , 0.98642534, 1.        ])
    # Coordinates:
    # * coh      (coh) float64 0.032 0.064 0.128 0.256 0.512