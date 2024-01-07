# Uses GitHub API to check test suite status (Travis, Jenkins)

import requests
import sys
from typing import Union

BASE_URL = "https://api.github.com/repos/brain-score/vision"


def get_data(url: str) -> dict:
    r = requests.get(url)
    assert r.status_code == 200
    return r.json()

def _get_end_time(d: dict) -> str:
    return d['end_time']

def _return_last_result(results: list) -> Union[str, None]:
    if results:
        last_result = max(results, key=_get_end_time)['result']
    else:
        last_result = None
    return last_result

def get_check_runs_result(run_name: str, check_runs_json: dict) -> str:
    check_runs = [{'end_time': run['completed_at'], 'result': run['conclusion']} 
                  for run in check_runs_json['check_runs'] if run['name'] == run_name]
    last_run_result = _return_last_result(check_runs)
    return last_run_result

def get_statuses_result(context: str, statuses_json: dict) -> str:
    statuses = [{'end_time': status['updated_at'], 'result': status['state']} 
                for status in statuses_json if status['context'] == context]
    last_status_result = _return_last_result(statuses)
    return last_status_result

def all_tests_passing(test_results: list):
    if any(result != "success" for result in test_results):
        return False
    else:
        return True


if __name__ == "__main__":

    pr_head_sha = sys.argv[1]

    check_runs_json = get_data(f"{BASE_URL}/commits/{pr_head_sha}/check-runs")
    statuses_json = get_data(f"{BASE_URL}/statuses/{pr_head_sha}")

    travis_branch_result = get_check_runs_result('Travis CI - Branch', check_runs_json)
    travis_pr_result = get_statuses_result('continuous-integration/travis', statuses_json)
    jenkins_plugintests_result = get_statuses_result('Brain-Score Jenkins CI - plugin tests', statuses_json)
    jenkins_unittests_result = get_statuses_result('Brain-Score Jenkins CI', statuses_json)

    print(all_tests_passing([travis_branch_result, travis_pr_result, jenkins_plugintests_result, jenkins_unittests_result]))
