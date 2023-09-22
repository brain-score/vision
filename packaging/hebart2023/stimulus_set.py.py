from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set
import pandas as pd
import re


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

for i in range(len(df)):
    # entire name of image file:
    image_id = df.iloc[i][0]
    image_paths[image_id] = stimulus_path + 'reference_images/' + df.iloc[i][1][1:]
    stimuli.append({
        'stimulus_id': image_id,
        'number': i
    })

print(stimuli)
stimuli = StimulusSet(stimuli)
stimuli.name = 'Hebart2023'  
stimuli.stimulus_paths = image_paths

# upload to S3
package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.name,
                     bucket_name="brainio-brainscore")