# Unit Tests
## Markers
Unit tests have various markers that denote possible issues in builds:

* **private_access**: tests that require access to a private ressource, such as assemblies on S3
* **memory_intense**: tests requiring more than 3 GB memory

Use the following syntax to mark a test:
```
@pytest.mark.memory_intense
def test_something(...):
    assert False
```

To skip a specific marker, run e.g. `pytest -m "not memory_intense"`.
To skip multiple markers, run e.g. `pytest -m "not private_access and not memory_intense"`.
