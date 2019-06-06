import pytest

private_access = pytest.mark.skipif(
    pytest.config.getoption("--skip-private"),
    reason="set to not run tests that require private s3 access")
memory_intense = pytest.mark.skipif(
    pytest.config.getoption("--skip-memory-intense"),
    reason="set to not run tests requiring a lot of memory (more than what travis offers)")
