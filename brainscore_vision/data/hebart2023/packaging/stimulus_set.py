from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set
import pandas as pd
import numpy as np

'''
Dataset Information:

- From Hebart 2023: https://elifesciences.org/articles/82580v1
- For metadata see https://osf.io/xtafs

- Images were taken from the THINGS object concept and image database.
- The datasset consists of 1,854 diverse object concepts sampled systematically 
  from concrete picturable and nameable nouns in the American English language.

Fields:
- image_number: a number {0,1...1853} indicates the image category
'''

stimuli = []
image_paths = {}
path = '/Users/linussommer/Desktop/Brain-Score-Data/StimulusSet/'
img_path = '/Users/linussommer/Desktop/Brain-Score-Data/StimulusSet/reference_images/'

df = pd.read_csv(path + 'metadata2.tsv', sep='\t')
df.drop(columns=['Word'], inplace=True)
df['image_no'] = df['image_no'].astype(str)

column_mapping = {
    # ids 
    'uniqueID': 'unique_id',
    'image_no': 'stimulus_id',
    'Example_image': 'example_image',
    
    # COCA
    'COCA_word_freq_online': 'word_freq_online',
    'COCA_word_freq': 'word_freq',
    'COCA_dispersion': 'dispersion',
    'COCA_rank': 'rank',
    
    # Frequencies
    'BNC_freq': 'freq_1',
    'SUBTLEX_freq': 'freq_2',
    
    # Hierarchical properties
    'Dominant_Part_of_Speech': 'dominant_part',
    'Bottom_up_Category_human': 'bottom_up',
    'Top_down_Category_WordNet': 'top_down_1',
    'Top_down_Category_manual': 'top_down_2',
    
    # WordNet
    'WordNet_Synonyms': 'WordNet_synonyms',
    'WordNet_ID': 'WordNet_ID',
    'Wordnet_ID2': 'Wordnet_ID2',
    'Wordnet_ID3': 'Wordnet_ID3',
    'Wordnet_ID4': 'Wordnet_ID4',
}

for index, row in df.iterrows():
    stimuli.append({column_mapping[col]: row[col] for col in df.columns})
    image_paths[row['image_no']] = img_path + row['uniqueID'] + '.jpg'

stimuli = StimulusSet(stimuli)
stimuli.name = 'Hebart2023'
stimuli.stimulus_paths = image_paths

assert len(stimuli) == 1854
assert len(stimuli.columns) == 18

package_stimulus_set("brainio_brainscore", stimuli,
                     stimulus_set_identifier=stimuli.name,
                     bucket_name="brainscore-storage/brainio-brainscore")
