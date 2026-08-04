import pytest
from subprocess import call

from brainscore_vision.submission.actions_helpers import BASE_URL, get_pr_num_from_head, get_data, get_statuses_result, are_all_tests_passing, is_labeled_automerge, get_pr_head_from_github_event

PR_HEAD_SHA = '47c6ee9089799a97d23e1aa813c49d8f12447fc0'
PR_BRANCH_NAME = 'patch-reduce-pnaslarge-mem'


def test_get_pr_num_from_head_pull_request(monkeypatch):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'pull_request')
    pr_num = get_pr_num_from_head(PR_BRANCH_NAME)
    assert pr_num == 2300

def test_get_pr_num_from_head_non_pull_request(monkeypatch):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'status')
    pr_num = get_pr_num_from_head(PR_HEAD_SHA)
    assert pr_num == 2300

def test_get_pr_head_status_event(monkeypatch, mocker):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'status')
    mock_status_json = {'branches': [{'name': 'master', 'commit': {'sha': 123}}, {'name': 'pr_branch', 'commit': {'sha': PR_HEAD_SHA}}]}
    mocker.patch('brainscore_vision.submission.actions_helpers._load_event_file', return_value=mock_status_json)
    assert get_pr_head_from_github_event() == PR_HEAD_SHA

def test_get_pr_head_status_event_master_only(monkeypatch, mocker):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'status')
    mock_status_json = {'branches': [{'name': 'master', 'commit': {'sha': 123}}]}
    mocker.patch('brainscore_vision.submission.actions_helpers._load_event_file', return_value=mock_status_json)
    assert not get_pr_head_from_github_event()

def test_get_pr_head_pull_request_event(monkeypatch):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'pull_request')
    monkeypatch.setenv('GITHUB_HEAD_REF', PR_BRANCH_NAME)
    assert get_pr_head_from_github_event() == PR_BRANCH_NAME

def test_get_statuses_result_len():
    data = get_data(f"{BASE_URL}/statuses/{PR_HEAD_SHA}")
    # GitHub's statuses endpoint accumulates entries every time a check is re-run,
    # so this asserts a non-empty list of well-formed entries rather than an exact count.
    assert len(data) >= 1
    assert all('context' in entry and 'state' in entry for entry in data)

def test_get_statuses_result():
    data = get_data(f"{BASE_URL}/statuses/{PR_HEAD_SHA}")
    jenkins_plugintests_result = get_statuses_result('Vision Unittests, Plugins', data)
    assert jenkins_plugintests_result == 'success'

def test_are_all_tests_passing():
    results_dict = {'jenkins_plugintests_result': 'success',
                'jenkins_unittests_result': 'success'}
    success = are_all_tests_passing(results_dict)
    assert success == True

def test_one_test_failing():
    results_dict = {'jenkins_plugintests_result': 'failure',
                    'jenkins_unittests_result': 'success'}
    success = are_all_tests_passing(results_dict)
    assert success == False
 
def test_is_labeled_automerge(mocker):
    assert is_labeled_automerge(1803) == True

def test_is_not_labeled_automerge(mocker):
    assert is_labeled_automerge(1801) == False

