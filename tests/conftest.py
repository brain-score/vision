def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption("--skip-private", action="store_true",
                     help="do not run tests requiring private s3 access")
    parser.addoption("--skip-gpu", action="store_true",
                     help="do not run tests requiring a GPU")
    parser.addoption("--skip-memory-intense", action="store_true",
                     help="do not run tests requiring a lot of memory")
