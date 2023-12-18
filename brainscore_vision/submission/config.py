import os


def get_database_secret() -> str:
    secret = os.getenv('BSC_DATABASESECRET')
    assert secret is not None, "Need to specify environment variable 'BSC_DATABASESECRET'"
    return secret
