from brainio.assemblies import DataAssembly
from brainscore_core import Metric, Score

from brainscore_vision.metric_helpers.transformations import apply_aggregate
from brainscore_vision.metrics import Ceiling


class InterSubjectConsistency(Ceiling):
    def __init__(self,
                 metric: Metric,
                 subject_column='subject'):
        """
        :param metric: The metric to compare two halves. Typically same as is used for model-data comparisons
        """
        self._metric = metric
        self._subject_column = subject_column

    def __call__(self, assembly: DataAssembly) -> Score:
        scores = []
        subjects = list(sorted(set(assembly[self._subject_column].values)))
        for target_subject in subjects:
            target_assembly = assembly[{'neuroid': [subject == target_subject
                                                    for subject in
                                                    assembly[self._subject_column].values]}]
            # predictor are all other subjects
            source_subjects = set(subjects) - {target_subject}
            pool_assembly = assembly[{'neuroid': [subject in source_subjects
                                                  for subject in
                                                  assembly[self._subject_column].values]}]
            score = self._metric(pool_assembly, target_assembly)
            # store scores
            score_has_subject = hasattr(score, 'raw') and hasattr(score.raw, self._subject_column)
            apply_raw = 'raw' in score.attrs and \
                        not score_has_subject  # only propagate if column not already part of score
            score = score.expand_dims(self._subject_column, _apply_raw=apply_raw)
            score.__setitem__(self._subject_column, [target_subject], _apply_raw=apply_raw)
            scores.append(score)

        scores = Score.merge(*scores)
        scores = apply_aggregate(lambda scores: scores.mean(self._subject_column), scores)
        return scores
