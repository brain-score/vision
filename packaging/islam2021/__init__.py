from fileinput import filename
from matplotlib.hatch import Shapes
import numpy as np
import os
import pandas as pd
from pathlib import Path
from brainio.stimuli import StimulusSet
from brainio.packaging import package_stimulus_set

#
# Code to generate the stylized voc 2012:
# https://github.com/islamamirul/shape_texture_neuron
#

shapes = ["aeroplane","bicycle","bird","boat","bottle","bus","car",
          "cat","chair","cow","diningtable","dog","horse","motorbike",
          "person","pottedplant","sheep","sofa","train","tvmonitor"]

textures = ["cracked","marbled","pleated","potholed","woven"]

#filename: t_s_y_id.jpg 
# t is texture in {0,1,2,3,4}
# s is shape in  {1,..,20}
# y is year
# i is a sequence of digits
# y_id identifies the original voc image
def collect_stylizedvoc_stimuli(data_dir):
    stimuli = []
    stimulus_paths = {}
    assert os.path.exists(data_dir)
    for filepath in Path(data_dir).glob('*.jpg'):
        assert len(filepath.stem.split('_')) == 4
        filename = filepath.name
        image_id = filepath.stem
        original_image_id = filepath.stem.split('_')[2] + "_" + filepath.stem.split('_')[3]  #corresponds to the original voc dataset
        texture_class = int(filepath.stem.split('_')[0])
        texture = textures[texture_class]
        shape_class = int(filepath.stem.split('_')[1]) - 1  #shape_class is the label in the original voc dataset.
        shape = shapes[shape_class]        
              
        stimulus_paths[image_id] = filepath
        stimuli.append({
            'stimulus_id': image_id,
            'original_image_id': original_image_id,
            'texture_class': texture_class,
            'texture': texture,
            'shape_class': shape_class, 
            'shape': shape
        })
            
    stimuli = StimulusSet(stimuli)
    stimuli.stimulus_paths = stimulus_paths
    stimuli.identifier = 'neil.Islam2021'
    assert len(stimuli) == 4369 * 5 
    return stimuli
    
    
if __name__ == '__main__':
    #dir = "brainscore/brain-score/packaging/stylizedvoc2012"
    dir = "./packaging/stylizedvoc2012"
    stimuli = collect_stylizedvoc_stimuli(dir)
    #print("finished collecting stimuli")
    # upload to S3
    #package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.identifier, bucket_name="brainio.contrib")
    #package_stimulus_set("brainio_brainscore", stimuli, stimulus_set_identifier=stimuli.identifier,bucket_name="brainio-brainscore")



