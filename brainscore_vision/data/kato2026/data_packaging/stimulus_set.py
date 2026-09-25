import os
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_core.supported_data_standards.brainio.packaging import package_stimulus_set_locally
import pandas as pd
conditions      = ['o','f','l','st','s','t','lf','cd','cs']
conditionLabels = ['Ori','Fil','Lin','SilTex','Sil','Tex','LinFil','Dense','Sparse']
imlist = sorted(os.listdir('stim'))
imlist = [x for x in imlist if 'ILSVRC2012' in x and 'dummy' not in x] # 240*7 (Exp1) + 171*2 (Exp2) =2022 images

for condition in conditions:
    stimuli_data = []
    imlist_condition = [x for x in imlist if x.endswith('_' + condition + '.jpg')]
    for im in imlist_condition:
        id = im.replace('.jpg', '')
        truth = 'kato2026_' + im.split('_')[0] # to avoid conflict with other benchmarks
        stimuli_data.append({'stimulus_id': id, 'truth': truth, 'condition': conditionLabels[conditions.index(condition)]})

    stimulus_set = StimulusSet(stimuli_data)
    stimulus_set.stimulus_paths = {stimulus['stimulus_id']: os.path.join('stim', imlist_condition[stimuli_data.index(stimulus)]) for stimulus in stimuli_data}
    stimulus_set.name = 'Kato2026.' + conditionLabels[conditions.index(condition)]
    
    # Packaging StimulusSet Locally
    package_info = package_stimulus_set_locally(
        proto_stimulus_set=stimulus_set,
        stimulus_set_identifier=stimulus_set.name,
        downloads_path=os.path.join('packaged')
    )
    if condition == conditions[0]:
        all_package_info = pd.DataFrame(columns=package_info.keys())
    all_package_info = pd.concat([all_package_info, pd.DataFrame([package_info])], ignore_index=True)

all_package_info.to_csv(os.path.join('packaged', 'stimulus_package_info.csv'), index=False)