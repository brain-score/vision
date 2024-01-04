# Unit Tests
## Markers
Unit tests have various markers that denote possible issues in the travis build.
The registered markers we use are listed in the `pyproject.toml` in `[tool.pytest.ini_options]`.

Use the following syntax to mark a test:
```
@pytest.mark.memory_intense
def test_something(...):
    assert False
```

To skip a specific marker, run e.g. `pytest -m "not memory_intense"`.
To skip multiple markers, run e.g. `pytest -m "not private_access and not memory_intense"`.


## Precomputed features
For many benchmark tests, it can be useful to evaluate on "reasonable" features that have been computed beforehand.
They are on S3 instead and are automatically downloaded by executing `bash test_setup.sh`.
Often these precomputed features are taken from models that were run on the benchmark.

To capture a model's activations, you can use the following steps:
1. in the benchmark, put a breakpoint right after `candidate.look_at`. 
   E.g. if you run a benchmark with `predictions = candidate.look_at(stimulus_set)`, capture the `predictions`
2. run the model you want on the benchmark, stopping at the breakpoint
3. store the model predictions to netcdf:
    ```python
    from brainio.packaging import write_netcdf
    
    write_netcdf(predictions, '<path/to/tests/<file>.nc')
    ```
4. upload the `.nc` file to the S3 brain-score-tests 
   [bucket](https://s3.console.aws.amazon.com/s3/buckets/brain-score-tests?region=us-east-1&prefix=tests/test_benchmarks/&showversions=false) 
   (drag and drop in browser is likely easiest; you might have to ask an admin to upload.)
   In the upload, make sure to make the file(s) publicly accessible: 
   under Permissions > Predefined ACLs > Grant public-read access
5. add filename to [`test_setup.sh`](https://github.com/brain-score/brain-score/blob/master/test_setup.sh) 
6. write your unit test (see e.g. 
   [here](https://github.com/brain-score/brain-score/blob/9ba55450a9d1c2b695c393df92aba2102ccdb169/tests/test_benchmarks/test_geirhos2021.py#L73) 
   for an example)
