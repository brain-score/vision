from brainscore_vision.submission.check_test_status import BASE_URL, get_data, get_check_runs_result, get_statuses_result, all_tests_passing

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
    success = all_tests_passing(['success', 'success', 'success', 'failure'])
    assert success == False

def test_all_tests_passing():
    success = all_tests_passing(['success', 'success', 'success', 'success'])
    assert success == True
