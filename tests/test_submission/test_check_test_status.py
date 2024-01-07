import pytest

from brainscore_vision.submission.check_test_status import BASE_URL, get_data, get_check_runs_result, get_statuses_result, are_all_tests_passing, is_labeled_automerge

PR_HEAD_SHA = '209e6c81d39179fd161a1bd3a5845682170abfd2'

def test_get_check_runs_data():
    data = get_data(f"{BASE_URL}/commits/{PR_HEAD_SHA}/check-runs")
    assert data['total_count'] == 6

def test_get_statuses_result():
    data = get_data(f"{BASE_URL}/statuses/{PR_HEAD_SHA}")
    assert len(data) == 9

def test_get_check_runs_result():
    data = get_data(f"{BASE_URL}/commits/{PR_HEAD_SHA}/check-runs")
    travis_branch_result = get_check_runs_result('Travis CI - Branch', data)
    assert travis_branch_result == 'success'

def test_get_statuses_result():
    data = get_data(f"{BASE_URL}/statuses/{PR_HEAD_SHA}")
    jenkins_plugintests_result = get_statuses_result('Brain-Score Jenkins CI - plugin tests', data)
    assert jenkins_plugintests_result == 'failure'

def test_one_test_failing():
    success = are_all_tests_passing(['success', 'success', 'success', 'failure'])
    assert success == False

def test_are_all_tests_passing():
    success = are_all_tests_passing(['success', 'success', 'success', 'success'])
    assert success == True
 
def test_is_labeled_automerge(mocker):
    dummy_check_runs_json = {"check_runs": [{"pull_requests": [{"url": "https://api.github.com/repos/brain-score/vision/pulls/453"}]}]}
    dummy_pull_request_data  = {"labels": [{"name": "automerge-web"}]}
    mocker.patch('brainscore_vision.submission.check_test_status.get_data', return_value=dummy_pull_request_data) 
    assert is_labeled_automerge(dummy_check_runs_json) == True

def test_is_not_labeled_automerge(mocker):
    dummy_check_runs_json = {"check_runs": [{"pull_requests": [{"url": "https://api.github.com/repos/brain-score/vision/pulls/453"}]}]}
    dummy_pull_request_data  = {'labels': []}
    mocker.patch('brainscore_vision.submission.check_test_status.get_data', return_value=dummy_pull_request_data) 
    assert is_labeled_automerge(dummy_check_runs_json) == False

def test_sha_associated_with_more_than_one_pr(mocker):
    dummy_check_runs_json = {"check_runs": [{"pull_requests": [{"url": "https://api.github.com/repos/brain-score/vision/pulls/453"}, {"url": "https://api.github.com/repos/brain-score/vision/pulls/452"}]}]}
    with pytest.raises(AssertionError):
        is_labeled_automerge(dummy_check_runs_json)


        
