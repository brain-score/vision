from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set
import pandas as pd

stimuli = []
image_paths = {}
stimulus_path = '/Users/linussommer/Desktop/Brain-Score-Data/StimulusSet/'

'''
Dataset Information:

- From Hebart 2023: https://elifesciences.org/articles/82580
- 1854 images in total

Fields:
- image_number: a number {0,1...1853} indicates the image category
'''

df = pd.read_csv(stimulus_path + 'metadata.csv')

for index, row in df.iterrows():
    image_id = row.iloc[0] 
    image_paths[image_id] = stimulus_path + 'reference_images/' + row.iloc[1][1:]
    stimuli.append({
        'stimulus_id': image_id,
        'number': index
    })

stimuli = StimulusSet(stimuli)
stimuli.name = 'Hebart2023'  
stimuli.stimulus_paths = image_paths

assert len(stimuli) == 1854

package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainio-brainscore")