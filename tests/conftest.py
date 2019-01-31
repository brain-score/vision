def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption("--skip-memory-intense", action="store_true",
                     help="do not run memory intense tests")
