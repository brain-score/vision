[![Build Status](https://travis-ci.com/brain-score/brain-score.svg?token=vqt7d2yhhpLGwHsiTZvT&branch=master)](https://travis-ci.com/brain-score/brain-score)
[![Documentation Status](https://readthedocs.org/projects/brain-score/badge/?version=latest)](https://brain-score.readthedocs.io/en/latest/?badge=latest)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) 

Brain-Score is a platform to evaluate computational models of brain function 
on their match to brain measurements in primate vision. 
The intent of Brain-Score is to adopt many (ideally all) the experimental benchmarks in the field
for the purpose of model testing, falsification, and comparison.
To that end, Brain-Score operationalizes experimental data into quantitative benchmarks 
that any model candidate following the [`BrainModel`](brainscore_vision/model_interface.py) interface can be scored on.

See the [Documentation](https://brain-score.readthedocs.io) for more details 
and the [Tutorial](https://brain-score.readthedocs.io/en/latest/modules/model_tutorial.html) 
and [Examples](https://github.com/brain-score/candidate_models/blob/master/examples/score-model.ipynb)
for submitting a model to Brain-Score.

Brain-Score is made by and for the community. 
To contribute, please [send in a pull request](https://github.com/brain-score/brain-score/pulls).


## Local installation

You will need Python = 3.7 and pip >= 18.1.
Note that you can only access public benchmarks when running locally.
To score a model on all benchmarks, submit it via the [brain-score.org website](http://www.brain-score.org).

`pip install git+https://github.com/brain-score/brain-score`

Score a model on a public benchmark:

```python
from brainscore_vision.benchmarks import public_benchmark_pool

benchmark = public_benchmark_pool['dicarlo.MajajHong2015public.IT-pls']
model = my_model()
score = benchmark(model)
# >  <xarray.Score (aggregation: 2)>
# >  array([0.32641998, 0.0207475])
# >  Coordinates:
# >    * aggregation  (aggregation) <U6 'center' 'error'
# >  Attributes:
# >      raw:                   <xarray.Score (aggregation: 2)>\narray([0.4278365 ...
# >      ceiling:               <xarray.Score (aggregation: 2)>\narray([0.7488407 ...
# >      model_identifier:      my-model
# >      benchmark_identifier:  dicarlo.MajajHong2015public.IT-pls
```

Some steps may take minutes because data has to be downloaded during first-time use.

For more details, see the [Documentation](https://brain-score.readthedocs.io) and 
the Examples [[1]](https://github.com/brain-score/brain-score/blob/master/examples) 
[[2]](https://github.com/brain-score/candidate_models/blob/master/examples).


## Environment Variables

| Variable               | Description                                                                                                                           |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| RESULTCACHING_HOME     | directory to cache results (benchmark ceilings) in, `~/.result_caching` by default (see https://github.com/brain-score/result_caching) |


## License

MIT license


## Troubleshooting

<details>
<summary>`ValueError: did not find HDF5 headers` during netcdf4 installation</summary>
pip seems to fail properly setting up the HDF5_DIR required by netcdf4.
Use conda: `conda install netcdf4`
</details>

<details>
<summary>repeated runs of a benchmark / model do not change the outcome even though code was changed</summary>
results (scores, activations) are cached on disk using https://github.com/mschrimpf/result_caching.
Delete the corresponding file or directory to clear the cache.
</details>


## CI environment

Add CI related build commands to `test_setup.sh`. The script is executed in CI environment for unittests.


## References

If you use Brain-Score in your work, please cite 
["Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like?"](https://www.biorxiv.org/content/10.1101/407007v2) (technical) and 
["Integrative Benchmarking to Advance Neurally Mechanistic Models of Human Intelligence"](https://www.cell.com/neuron/fulltext/S0896-6273(20)30605-X) (perspective) 
as well as the respective benchmark sources.

```bibtex
@article{SchrimpfKubilius2018BrainScore,
  title={Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like?},
  author={Martin Schrimpf and Jonas Kubilius and Ha Hong and Najib J. Majaj and Rishi Rajalingham and Elias B. Issa and Kohitij Kar and Pouya Bashivan and Jonathan Prescott-Roy and Franziska Geiger and Kailyn Schmidt and Daniel L. K. Yamins and James J. DiCarlo},
  journal={bioRxiv preprint},
  year={2018},
  url={https://www.biorxiv.org/content/10.1101/407007v2}
}

@article{Schrimpf2020integrative,
  title={Integrative Benchmarking to Advance Neurally Mechanistic Models of Human Intelligence},
  author={Schrimpf, Martin and Kubilius, Jonas and Lee, Michael J and Murty, N Apurva Ratan and Ajemian, Robert and DiCarlo, James J},
  journal={Neuron},
  year={2020},
  url={https://www.cell.com/neuron/fulltext/S0896-6273(20)30605-X}
}
```

