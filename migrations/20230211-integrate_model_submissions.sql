-- unique submission ids of all models with a score
select distinct(bs.id)
from brainscore_score
         join brainscore_model bm on brainscore_score.model_id = bm.id
         join brainscore_submission bs on bm.submission_id = bs.id;
