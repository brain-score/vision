import pytest
from subprocess import call

from brainscore_vision.submission.actions_helpers import BASE_URL, get_pr_num_from_head, get_data, get_check_runs_result, get_statuses_result, are_all_tests_passing, is_labeled_automerge, get_pr_head_from_github_event

pr_head_sha = '209e6c81d39179fd161a1bd3a5845682170abfd2'
pr_branch_name = 'web_submission_11/add_plugins'


def test_get_pr_num_from_head_pull_request(monkeypatch):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'pull_request')
    pr_num = get_pr_num_from_head(pr_branch_name)
    assert pr_num == 442

def test_get_pr_num_from_head_non_pull_request(monkeypatch):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'status')
    pr_num = get_pr_num_from_head(pr_head_sha)
    assert pr_num == 442

def test_get_pr_head_status_event(monkeypatch, mocker):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'status')
    mock_status_json = {'branches': [{'name': 'master', 'commit': {'sha': 123}}, {'name': 'pr_branch', 'commit': {'sha': pr_head_sha}}]}
    mocker.patch('brainscore_vision.submission.actions_helpers._load_event_file', return_value=mock_status_json)
    assert get_pr_head_from_github_event() == pr_head_sha

def test_get_pr_head_status_event_master_only(monkeypatch, mocker):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'status')
    mock_status_json = {'branches': [{'name': 'master', 'commit': {'sha': 123}}]}
    mocker.patch('brainscore_vision.submission.actions_helpers._load_event_file', return_value=mock_status_json)
    assert not get_pr_head_from_github_event()

def test_get_pr_head_check_run_event(monkeypatch, mocker):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'check_run')
    mock_check_run_json = {'check_run': {'head_sha': pr_head_sha}}
    mocker.patch('brainscore_vision.submission.actions_helpers._load_event_file', return_value=mock_check_run_json)
    assert get_pr_head_from_github_event() == pr_head_sha

def test_get_pr_head_pull_request_event(monkeypatch):
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'pull_request')
    monkeypatch.setenv('GITHUB_HEAD_REF', pr_branch_name)
    assert get_pr_head_from_github_event() == pr_branch_name

def test_get_check_runs_data():
    data = get_data(f"{BASE_URL}/commits/{pr_head_sha}/check-runs")
    assert data['total_count'] == 6

def test_get_statuses_result():
    data = get_data(f"{BASE_URL}/statuses/{pr_head_sha}")
    assert len(data) == 9

def test_get_check_runs_result():
    data = get_data(f"{BASE_URL}/commits/{pr_head_sha}/check-runs")
    travis_branch_result = get_check_runs_result('Travis CI - Branch', data)
    assert travis_branch_result == 'success'

def test_get_statuses_result():
    data = get_data(f"{BASE_URL}/statuses/{pr_head_sha}")
    jenkins_plugintests_result = get_statuses_result('Brain-Score Jenkins CI - plugin tests', data)
    assert jenkins_plugintests_result == 'failure'

def test_are_all_tests_passing():
    results_dict = {'travis_branch_result': 'success',
                'travis_pr_result': 'success',
                'jenkins_plugintests_result': 'success',
                'jenkins_unittests_result': 'success'}
    success = are_all_tests_passing(results_dict)
    assert success == True

def test_one_test_failing():
    results_dict = {'travis_branch_result': 'success',
                    'travis_pr_result': 'failure',
                    'jenkins_plugintests_result': 'success',
                    'jenkins_unittests_result': 'success'}
    success = are_all_tests_passing(results_dict)
    assert success == False
 
def test_is_labeled_automerge(mocker):
    dummy_check_runs_json = {"check_runs": [{"pull_requests": [{"url": "https://api.github.com/repos/brain-score/vision/pulls/453"}]}]}
    dummy_pull_request_data  = {"labels": [{"name": "automerge-web"}]}
    mocker.patch('brainscore_vision.submission.actions_helpers.get_data', return_value=dummy_pull_request_data) 
    assert is_labeled_automerge(dummy_check_runs_json) == True

def test_is_not_labeled_automerge(mocker):
    dummy_check_runs_json = {"check_runs": [{"pull_requests": [{"url": "https://api.github.com/repos/brain-score/vision/pulls/453"}]}]}
    dummy_pull_request_data  = {'labels': []}
    mocker.patch('brainscore_vision.submission.actions_helpers.get_data', return_value=dummy_pull_request_data) 
    assert is_labeled_automerge(dummy_check_runs_json) == False

def test_sha_associated_with_more_than_one_pr():
    dummy_check_runs_json = {"check_runs": [{"pull_requests": [{"url": "https://api.github.com/repos/brain-score/vision/pulls/453"}, {"url": "https://api.github.com/repos/brain-score/vision/pulls/452"}]}]}
    with pytest.raises(AssertionError):
        is_labeled_automerge(dummy_check_runs_json)

        
