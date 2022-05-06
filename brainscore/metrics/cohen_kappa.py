from brainscore.metrics import Metric


class CohensKappa(Metric):
    """
    Computes the error consistency using Cohen's Kappa.
    Cohen, 1960 https://doi.org/10.1177%2F001316446002000104
    implemented in Bethge et al., 2021 https://proceedings.neurips.cc/paper/2021/hash/c8877cff22082a16395a57e97232bb6f-Abstract.html
    """

    def __init__(self, collapse_distractors, normalize=False, repetitions=2):
        super().__init__()
        self._collapse_distractors = collapse_distractors
        self._normalize = normalize
        self._repetitions = repetitions
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source_probabilities, target):
        # from https://github.com/bethgelab/model-vs-human/blob/745046c4d82ff884af618756bd6a5f47b6f36c45/modelvshuman/plotting/analyses.py#L161

        p1 = SixteenClassAccuracy().analysis(df1)["16-class-accuracy"]
        p2 = SixteenClassAccuracy().analysis(df2)["16-class-accuracy"]
        expected_consistency = p1 * p2 + (1 - p1) * (1 - p2)

        df1["is_correct"] = df1.object_response == df1.category
        df2["is_correct"] = df2.object_response == df2.category
        observed_consistency = (df1.is_correct == df2.is_correct).sum() / len(df1)

        error_consistency = self.error_consistency(expected_consistency=expected_consistency,
                                                   observed_consistency=observed_consistency)

        return self._repeat(lambda random_state:
                            self._call_single(source_probabilities, target, random_state=random_state))

    def error_consistency(self, expected_consistency, observed_consistency):
        # from https://github.com/bethgelab/model-vs-human/blob/745046c4d82ff884af618756bd6a5f47b6f36c45/modelvshuman/plotting/analyses.py#L147-L158
        """Return error consistency as measured by Cohen's kappa."""

        assert expected_consistency >= 0.0
        assert expected_consistency <= 1.0
        assert observed_consistency >= 0.0
        assert observed_consistency <= 1.0

        if observed_consistency == 1.0:
            return 1.0
        else:
            return (observed_consistency - expected_consistency) / (1.0 - expected_consistency)
