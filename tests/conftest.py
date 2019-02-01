def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption("--skip-private", action="store_true",
                     help="do not run tests requiring private s3 access")
