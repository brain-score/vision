# Uses GitHub API to check test suite status (Travis, Jenkins)

import requests
import sys

BASE_URL = "https://api.github.com/repos/brain-score/vision"


def get_data(url: str) -> dict:
    r = requests.get(url)
    assert r.status_code == 200
    return r.json()

def get_check_runs_result(run_name: str, check_runs: dict) -> str:
    last_run_result = next(run['conclusion'] for run in check_runs['check_runs'] if run['name'] == run_name)
    return last_run_result

def get_statuses_result(context: str, statuses: dict) -> str:
    last_status_result = next(status['state'] for status in statuses if status['context'] == context)
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
