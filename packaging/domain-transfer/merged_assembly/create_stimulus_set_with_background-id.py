# import libraries 
## general libraries 
import pandas as pd

## brain-score libraries
import brainscore

## domain-transfer specific libraries
from helpers_background_id import loading_brain_model_activation


##### NOTES ON THIS SCRIPT #####
# This script is used to create a stimulus set with the background-id included. 
# A fair question is: why do we need a model to do this? Shouldn't it depend only on the stimulus set?
# the answer is: yes, it should. But this implementation takes the already existing from previous work 
# of Marliawaty I Gusti Bagus, where the background is created as the presentation dimension of the 
# brain activations extracted using the 'look_at' function from the brain-score package that is called 
# on a stimulus set that doesn't have the background-id included.
# It is however independent of the model used to extract brain activations.
# FUTURE WORK should only use the stimulus set to create the background-id, and not the model activations.
#################################

# get the background-id
brain_model_name='dcgan'
image_source='dicarlo.domain_transfer'

domain_transfer_data = loading_brain_model_activation(brain_model_name, image_source, penultimate_layer=False)

background_df = pd.DataFrame({'id': domain_transfer_data.stimulus_id.values, 'background': domain_transfer_data.background_id.values})

# include the background-id in the stimulus set
stimulus_set = brainscore.get_stimulus_set('dicarlo.domain_transfer')

dict_background = {}
for row in range(len(background_df)):
    dict_background[background_df.id[row]] = background_df.background[row]

stimulus_set['background_id'] = [0] * len(stimulus_set)

for i in range(len(stimulus_set)):
    if stimulus_set.stimulus_id[i] in dict_background:
        stimulus_set['background_id'][i] = int(dict_background[stimulus_set.stimulus_id[i]])


stimulus_set.to_csv('merged_stimulus_set.csv', index=False)