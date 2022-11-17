from pathlib import Path

import fire
from brainscore_core.plugin_management.test_plugins import run_args as _core_test_plugins

from brainscore_vision import score as _score_function


def score(model_identifier: str, benchmark_identifier: str):
    result = _score_function(model_identifier, benchmark_identifier)
    print(result)  # print instead of return because fire has issues with xarray objects


def test_plugins(*args, **kwargs):
    root_directory = Path(__file__).parent
    _core_test_plugins(*args, root_directory=root_directory, **kwargs)


if __name__ == '__main__':
    fire.Fire()
