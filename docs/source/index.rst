Brain-Score
===========

Brain-Score is a collection of benchmarks:
combinations of data and metrics that score any model on how brain-like it is.

Data is organized in BrainIO_,
metrics and benchmarks are implemented in this repository,
and standard models are implemented in candidate-models_.

.. _BrainIO: https://github.com/brain-score/brainio
.. _candidate-models: https://github.com/brain-score/candidate_models

The primary method this library provides is the `score_model` function.

.. autofunction:: brainscore.score_model

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   modules/model_tutorial
   modules/benchmark_tutorial
   modules/model_interface
   modules/benchmarks
   modules/metrics
   modules/submission
   modules/utils
   modules/api_reference

