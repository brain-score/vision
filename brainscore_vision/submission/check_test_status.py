# Uses GitHub API to check test suite status (Travis, Jenkins)

import json
import os
import requests
import sys
from typing import Union

BASE_URL = "https://api.github.com/repos/brain-score/vision"


def _load_event_file() -> dict:
    with open(os.environ["GITHUB_EVENT_PATH"]) as f:
        return json.load(f)

def get_pr_head_sha() -> Union[str, None]:
    event_type = os.environ["GITHUB_EVENT_NAME"]
    pr_head_sha = None

    if event_type == "status":
        f = _load_event_file()
        candidate_branches = [branch for branch in f["branches"] if branch["name"] != "master"]
        if len(candidate_branches) == 1:
            pr_head_sha = candidate_branches[0]["commit"]["sha"]

    elif event_type == "check_run":
        f = _load_event_file()
        pr_head_sha = f["head_sha"] 

    elif event_type == "pull_request":
        pr_head_sha = os.environ["GITHUB_HEAD_REF"]

    return pr_head_sha

def print_pr_head_sha():
    print(get_pr_head_sha()) # for logging in action

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

def are_all_tests_passing(test_results: list):
    if any(result != "success" for result in test_results):
        return False
    else:
        return True
    
def is_labeled_automerge(check_runs_json: dict) -> bool:
    pull_requests = [check_run['pull_requests'] for check_run in check_runs_json['check_runs']]
    assert all(len(pull_request) == 1 for pull_request in pull_requests), f'Expected one PR associated with this SHA but found none or more than one, cannot automerge'
    pull_request_data = get_data(pull_requests[0][0]["url"])
    labeled_automerge = any(label['name'] in ('automerge', 'automerge-web') for label in pull_request_data['labels'])
    return labeled_automerge


if __name__ == "__main__":

    pr_head_sha = get_pr_head_sha()
    if not pr_head_sha:
        print(False)
        sys.exit()

    check_runs_json = get_data(f"{BASE_URL}/commits/{pr_head_sha}/check-runs")
    statuses_json = get_data(f"{BASE_URL}/statuses/{pr_head_sha}")

    travis_branch_result = get_check_runs_result('Travis CI - Branch', check_runs_json)
    travis_pr_result = get_statuses_result('continuous-integration/travis', statuses_json)
    jenkins_plugintests_result = get_statuses_result('Brain-Score Jenkins CI - plugin tests', statuses_json)
    jenkins_unittests_result = get_statuses_result('Brain-Score Jenkins CI', statuses_json)

    tests_pass = are_all_tests_passing([travis_branch_result, travis_pr_result, jenkins_plugintests_result, jenkins_unittests_result])

    if tests_pass:
        print(is_labeled_automerge(check_runs_json))
    else:
        print(False)
