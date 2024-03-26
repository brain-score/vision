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
import smtplib
from typing import Union
from email.mime.text import MIMEText

BASE_URL = "https://api.github.com/repos/brain-score/vision"


def get_data(url: str) -> dict:
    r = requests.get(url)
    assert r.status_code == 200, f'{r.status_code}: {r.reason}'
    return r.json()

def get_pr_num_from_head(pr_head) -> int:
    """
    Given either an SHA (status event) or a branch name (pull request event),
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
    Based on the event that triggered the workflow (status or pull_request),
    returns either an SHA (status) or branch name (pull_request)
    """
    pr_head = None
    event_type = os.environ["GITHUB_EVENT_NAME"]

    if event_type == "status":
        f = _load_event_file()
        candidate_branches = [branch for branch in f["branches"] if branch["name"] != "master"]
        if len(candidate_branches) == 1:
            pr_head = candidate_branches[0]["commit"]["sha"]

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
        
def any_tests_failing(test_results: dict) -> dict:
    if any(result == "failure" for result in test_results.values()):
        return True
    else:
        return False
    
def is_labeled_automerge(pr_num: int) -> bool:
    label_data = get_data(f"{BASE_URL}/issues/{pr_num}/labels")
    labeled_automerge = any(label['name'] in ('automerge', 'automerge-web') for label in label_data)
    return labeled_automerge

def send_failure_email(email: str, pr_number: str, mail_username: str, mail_password: str):
    """ Send submitter an email if their web-submitted PR fails. """
    body = "Your Brain-Score submission did not pass checks. " \
           "Please review the test results and update the PR at " \
           f"https://github.com/brain-score/vision/pull/{pr_number} " \
           "or send in an updated submission via the website."
    msg = MIMEText(body)
    msg['Subject'] = "Brain-Score submission failed"
    msg['From'] = "Brain-Score"
    msg['To'] = email

    # send email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
        smtp_server.login(mail_username, mail_password)
        smtp_server.sendmail(mail_username, email, msg.as_string())


if __name__ == "__main__":
    
    pr_head = get_pr_head_from_github_event()
    if not pr_head:
        print("No PR head found. Exiting."); sys.exit()
    pr_num = get_pr_num_from_head(pr_head)
    
    # GitHub Actions helpers
    if len(sys.argv) > 1:
        if sys.argv[1] == "get_pr_head":
            print(pr_head)
        elif sys.argv[1] == "get_pr_num":
            print(pr_num)
        sys.exit()

    # Check test results and ensure PR is automergeable
    statuses_json = get_data(f"{BASE_URL}/statuses/{pr_head}")

    results_dict = {'travis_pr_result': get_statuses_result('continuous-integration/travis', statuses_json),
                    'jenkins_plugintests_result': get_statuses_result('Brain-Score Jenkins CI - plugin tests', statuses_json),
                    'jenkins_unittests_result': get_statuses_result('Brain-Score Jenkins CI', statuses_json)}

    tests_pass = are_all_tests_passing(results_dict)
    tests_fail = any_tests_failing(results_dict)

    if tests_pass:
        if is_labeled_automerge(pr_num):
            print(True)
        else:
            print("All tests pass but not labeled for automerge. Exiting.")
    else:
        if tests_fail:
            if is_labeled_automerge(pr_num):
                print("Failure")
        print(results_dict)
