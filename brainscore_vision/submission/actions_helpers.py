"""
Supports two GitHub Actions automerge workflows:
1. automerge_plugin-only_prs
2. check_if_pr_is_automergeable
Uses GitHub API to support the following functions
- Retrieve a PR Head (SHA or branch name) from the workflow trigger event
- Retrieve a PR number from a PR Head (SHA or branch name)
- Check test suite status (Travis, Jenkins)
Final outputs are accessed by the action via print()
"""

import json
import os
import requests
import sys
from typing import Union

BASE_URL = "https://api.github.com/repos/brain-score/vision"


def get_data(url: str) -> dict:
    r = requests.get(url)
    assert r.status_code == 200, f'{r.status_code}: {r.reason}'
    return r.json()

def get_pr_num_from_head(pr_head) -> int:
    """
    Given either an SHA (check_run or status event) or a branch name (pull request event),
    returns the number of the pull request with that head SHA or branch name.
    """
    event_type = os.environ["GITHUB_EVENT_NAME"]

    if event_type == "pull_request":
        query = f"repo:brain-score/vision type:pr head:{pr_head}"
    else:
        query = f"repo:brain-score/vision type:pr sha:{pr_head}"
    url = f"https://api.github.com/search/issues?q={query}"
    pull_requests = get_data(url)
    assert pull_requests["total_count"] == 1, f'Expected one PR associated with this SHA but found none or more than one, cannot automerge'
    pr_num = pull_requests["items"][0]["number"]

    return pr_num

def _load_event_file() -> dict:
    with open(os.environ["GITHUB_EVENT_PATH"]) as f:
        return json.load(f)
    
def get_pr_head_from_github_event() -> str:
    """
    Based on the event that triggered the workflow (check_run, status, or pull_request),
    returns either an SHA (check_run or status) or branch name (pull_request)
    """
    pr_head = None
    event_type = os.environ["GITHUB_EVENT_NAME"]

    if event_type == "status":
        f = _load_event_file()
        candidate_branches = [branch for branch in f["branches"] if branch["name"] != "master"]
        if len(candidate_branches) == 1:
            pr_head = candidate_branches[0]["commit"]["sha"]

    elif event_type == "check_run":
        f = _load_event_file()
        pr_head = f["check_run"]["head_sha"]

    elif event_type == "pull_request":
        pr_head = os.environ["GITHUB_HEAD_REF"]

    return pr_head

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

def are_all_tests_passing(test_results: dict) -> dict:
    if any(result != "success" for result in test_results.values()):
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
    
    pr_head = get_pr_head_from_github_event()
    if not pr_head:
        print("No PR head found. Exiting."); sys.exit()
    
    # GitHub Actions helpers
    if len(sys.argv) > 1:
        if sys.argv[1] == "get_pr_head":
            print(pr_head)
        elif sys.argv[1] == "get_pr_num":
            print(get_pr_num_from_head(pr_head))
        sys.exit()

    # Check test results and ensure PR is automergeable
    check_runs_json = get_data(f"{BASE_URL}/commits/{pr_head}/check-runs")
    statuses_json = get_data(f"{BASE_URL}/statuses/{pr_head}")

    results_dict = {'travis_branch_result': get_check_runs_result('Travis CI - Branch', check_runs_json),
                    'travis_pr_result': get_statuses_result('continuous-integration/travis', statuses_json),
                    'jenkins_plugintests_result': get_statuses_result('Brain-Score Jenkins CI - plugin tests', statuses_json),
                    'jenkins_unittests_result': get_statuses_result('Brain-Score Jenkins CI', statuses_json)}

    tests_pass = are_all_tests_passing(results_dict)

    if tests_pass:
        if is_labeled_automerge(check_runs_json):
            print(True)
        else:
            print("All tests pass but not labeled for automerge. Exiting.")
    else:
        print(results_dict)
