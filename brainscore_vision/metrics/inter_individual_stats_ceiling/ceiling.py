from xarray import DataArray

from brainscore_vision.metric_helpers.transformations import apply_aggregate
from brainscore_vision import Ceiling
from brainscore_vision import Score


class InterIndividualStatisticsCeiling(Ceiling):
    """
    Cross-validation-like, animal-wise computation of ceiling
    """

    def __init__(self, metric):
        """
        :param metric: used to compute the ceiling
        """
        self._metric = metric

    def __call__(self, statistic: DataArray) -> Score:
        """
        Applies given metric to dataset, comparing data from one animal to all remaining animals, i.e.:
        For each animal: metric({dataset\animal_i}, animal_i): cross validation like
        :param statistic: xarray structure with values & and corresponding meta information: distances, source
        :return: ceiling
        """
        assert len(set(statistic.source.data)) > 1, 'your stats contain less than 2 subjects'
        self.statistic = statistic

        monkey_scores = []
        for heldout_monkey in sorted(set(self.statistic.source.data)):
            monkey_pool = self.statistic.where(self.statistic.source != heldout_monkey, drop=True)
            heldout = self.statistic.sel(source=heldout_monkey)
            score = self._metric(monkey_pool, heldout)

            score = score.expand_dims('monkey')
            score['monkey'] = [heldout_monkey]
            monkey_scores.append(score)
        # aggregate
        scores = Score.merge(*monkey_scores)
        return apply_aggregate(lambda s: s.mean('monkey'), scores)

        # Note: this should probably be more general for arbitrary subjects instead of 'monkey'
