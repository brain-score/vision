from brainscore_vision.submission import config
test_database = 'brainscore-ohio-test-migrated'
config.get_database_secret = lambda: test_database
