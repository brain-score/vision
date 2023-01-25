# Unit Tests
## Markers
Unit tests have various markers that denote possible issues in the travis build:

* **private_access**: tests that require access to a private ressource, such as assemblies on S3 (travis pull request builds can not have private access)
* **memory_intense**: tests requiring more memory than is available in the travis sandbox (currently 3 GB, https://docs.travis-ci.com/user/common-build-problems/#my-build-script-is-killed-without-any-error)
* **requires_gpu**: tests requiring a GPU to run or to run in a reasonable time (travis does not support GPUs/CUDA)
* **slow**: tests leading to runtimes that are not possible on the openmind cluster (>1 hour per test) 
* **travis_slow**: tests running for more than 10 minutes without output (which leads travis to error)

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
1. run the model you want on the benchmark
2. locate the cached (pickled) activations 
   (likely in `~/.result_caching/model_tools.activations.core.ActionsExtractorHelper._from_paths_stored/<filename>.pkl`)
3. convert pickled activations into netcdf:
    ```python
    import pickle
    from brainio.packaging import write_netcdf
    
    with open('~/.result_caching/.../<filename>.pkl', 'rb') as f:
        pickled_data = pickle.load(f)
        activations = pickled_data['data']
        write_netcdf(activations, '<path/to/tests/<file>.nc')
    ```
4. upload the `.nc` file to the S3 brain-score-tests 
   [bucket](https://s3.console.aws.amazon.com/s3/buckets/brain-score-tests?region=us-east-1&prefix=tests/test_benchmarks/&showversions=false) 
   (drag and drop in browser is likely easiest; you might have to ask an admin to upload)
5. add filename to [`test_setup.sh`](https://github.com/brain-score/brain-score/blob/master/test_setup.sh) 
6. write your unit test (see e.g. 
   [here](https://github.com/brain-score/brain-score/blob/9ba55450a9d1c2b695c393df92aba2102ccdb169/tests/test_benchmarks/test_geirhos2021.py#L73) 
   for an example)
