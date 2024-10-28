# Created by David Coggan on 2024 06 25

from brainscore_vision import benchmark_registry
from .benchmark import (
    Coggan2024_behavior_ConditionWiseAccuracySimilarity)

benchmark_registry['tong.Coggan2024_behavior-ConditionWiseAccuracySimilarity'] = (
    Coggan2024_behavior_ConditionWiseAccuracySimilarity)

