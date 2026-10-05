from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
from brainscore_core.supported_data_standards.brainio import packaging
import pandas as pd
import numpy as np
import os

conditionLabels = ['Ori','Fil','Lin','SilTex','Sil','Tex','LinFil','Dense','Sparse']
for condition in conditionLabels:
    csv_path = os.path.join('raw', f'{condition}.csv')
    df = pd.read_csv(csv_path)
    behavioral_data = df['response'].values  # Assuming 'response' column exists
    assembly = BehavioralAssembly(
        behavioral_data,
        coords={
            'stimulus_id': ('presentation',df['stimulus_id'].astype(str)),
            'subject': ('presentation', df['subject'].astype(str)),
            'correct': ('presentation', df['correct'].values),
            'condition': ('presentation', np.repeat(conditionLabels[conditionLabels.index(condition)], len(df))),
            'response': ('presentation', df['response'].values),
            'truth': ('presentation', df['truth'].values)
        },
        dims=['presentation']
    )
    
    stimsetname = 'Kato2026.' + conditionLabels[conditionLabels.index(condition)]
    package_info = packaging.package_data_assembly_locally(
        proto_data_assembly=assembly,
        assembly_identifier=stimsetname, # We use the same stimulusSet name for the assembly
        stimulus_set_identifier=stimsetname,
        assembly_class_name="BehavioralAssembly", # For most neural data, use NeuroidAssembly. For behavioral data, use BehavioralAssembly
        downloads_path=os.path.join('packaged')        
    )
    if condition == conditionLabels[0]:
        all_package_info = pd.DataFrame(columns=package_info.keys())
    all_package_info = pd.concat([all_package_info, pd.DataFrame([package_info])], ignore_index=True)
all_package_info.to_csv(os.path.join('packaged', 'assembly_package_info.csv'), index=False)
    
    