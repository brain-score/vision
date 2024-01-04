from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments.cornets import cornet_brain_pool
from candidate_models.model_commitments.vonecornets import vonecornet_brain_pool
from candidate_models.model_commitments.model_layer_def import model_layers
from brainscore.submission.ml_pool import MLBrainPool
from brainscore.submission.utils import UniqueKeyDict

brain_translated_pool = UniqueKeyDict(reload=True)

ml_brain_pool = MLBrainPool(base_model_pool, model_layers)

for identifier, model in ml_brain_pool.items():
    brain_translated_pool[identifier] = model

for identifier, model in cornet_brain_pool.items():
    brain_translated_pool[identifier] = model

for identifier, model in vonecornet_brain_pool.items():
    brain_translated_pool[identifier] = model
