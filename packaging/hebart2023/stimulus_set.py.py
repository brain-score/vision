from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set
import pandas as pd

stimuli = []
image_paths = {}
stimulus_path = '/Users/linussommer/Desktop/Brain-Score-Data/StimulusSet/'

'''
Dataset Information:

- From Hebart 2019: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223792

- Images were taken from the THINGS object concept and image database.
- The datasset consists of 1,854 diverse object concepts sampled systematically 
  from concrete picturable and nameable nouns in the American English language.

Fields:
- image_number: a number {0,1...1853} indicates the image category
'''

# initial csv to dataframe processing:
df = pd.read_csv(stimulus_path + 'metadata.csv')

# add image_ids to correspond with the experimental data
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