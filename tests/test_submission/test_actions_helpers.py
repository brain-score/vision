import pytest
from subprocess import call

from brainscore_vision.submission.actions_helpers import BASE_URL, get_pr_num_from_head, get_data, get_statuses_result, are_all_tests_passing, is_labeled_automerge, get_pr_head_from_github_event

PR_HEAD_SHA = '6ff4c26e35f67ad13bd075dbb73bd5166c854b7d'
PR_BRANCH_NAME = 'web_submission_741/add_plugins'


def test_get_pr_num_from_head_pull_request(monkeypatch):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'pull_request')
    pr_num = get_pr_num_from_head(PR_BRANCH_NAME)
    assert pr_num == 1803

def test_get_pr_num_from_head_non_pull_request(monkeypatch):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'status')
    pr_num = get_pr_num_from_head(PR_HEAD_SHA)
    assert pr_num == 1803

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
    assert len(data) == 8

def test_get_statuses_result():
    data = get_data(f"{BASE_URL}/statuses/{PR_HEAD_SHA}")
    jenkins_plugintests_result = get_statuses_result('Brain-Score Plugins Unit tests (AWS Jenkins, AWS Execution)', data)
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

