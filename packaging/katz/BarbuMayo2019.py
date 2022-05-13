from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

import csv
import pandas as pd
stimuli = []  # collect meta
image_paths = {}  # collect mapping of image_id to filepath
stimuli_directory = "images_no_border"

with open('mappings/objectnet.csv' ,'r') as f:
    df = pd.read_csv(f)
    df = df.reset_index(False)
    df = df.rename(columns={"index":"image_id", "imagenet_label":"synset", "file_name":"filepath"})
    df = df.astype({'image_id':'str'})
    df['filepath'] = 'images_no_border/'+df['filepath']
    #print(df)
print(df.dtypes)
stimuli = StimulusSet(df)
stimuli.image_paths = {row.image_id: row.filepath for row in stimuli.itertuples()}

stimuli.name = 'katz.BarbuMayo2019'  # give the StimulusSet an identifier name

assert len(stimuli) == 17261  # make sure the StimulusSet is what you would expect

package_stimulus_set(catalog_name="brainio_brainscore", proto_stimulus_set=stimuli, stimulus_set_identifier=stimuli.name, bucket_name="brainio-brainscore")  # upload to S3