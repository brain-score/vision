from brainscore_vision import metric_registry
from .ceiling import InterSubjectConsistency

metric_registry['inter_subject_consistency'] = InterSubjectConsistency
